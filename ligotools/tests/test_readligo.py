import unittest
from unittest.mock import patch
import numpy as np
import os
import sys

# FIX: Adjust the path to allow 'readligo' module discovery.
# We assume readligo.py is in the parent directory of the test file's location.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import readligo
from readligo import SegmentList, loaddata, read_hdf5, dq_channel_to_seglist

# ====================================================================
# A. Mocks and Fixtures
# ====================================================================

# Mock class for os.stat to prevent the "zero length file" check from failing
class MockStat:
    st_size = 100 

# Mock implementation of read_hdf5 to simulate data reading without h5py
def mock_read_hdf5_impl(filename, readstrain):
    """Simulates reading a 4-second (1Hz sampled) HDF5 file."""
    if readstrain:
        strain = np.array([1.0, 2.0, 3.0, 4.0])
    else:
        strain = 0
    gpsStart = 123456789.0
    ts = 1.0 # 1 Hz sample time for strain (simplifying for test)
    
    # DQ mask: 0b11 (3) for first 2 points, 0b00 (0) after
    qmask = np.array([3, 3, 0, 0], dtype='int32') 
    shortnameList = ['DATA', 'SCIENCE'] # bit 0=DATA, bit 1=SCIENCE
    
    # INJ mask: 0b01 (1) for first 2 points, 0b00 (0) after
    injmask = np.array([1, 1, 0, 0], dtype='int32') 
    injnameList = ['HW_INJ'] # bit 0=HW_INJ
    
    return strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList
        
# ====================================================================
# B. Corrected Test Functions
# ====================================================================

# Test 1: loaddata (HDF5 path)
# We patch 'readligo.read_hdf5' because 'loaddata' (in readligo.py) calls it.
@patch('os.stat', return_value=MockStat())
@patch('readligo.read_hdf5', side_effect=mock_read_hdf5_impl)
def test_loaddata_hdf5(mock_read_hdf5, mock_stat):
    """
    Test loaddata functionality for HDF5 files, ensuring time vector and 
    channel dictionary are correctly created from mocked data.
    """
    
    # --- Simulate loaddata call with tvec=True ---
    strain, time_vec, channel_dict = loaddata('test.hdf5', ifo='H1', tvec=True)
    
    # 1. Assert HDF5 reader was called
    assert mock_read_hdf5.called == True
    
    # 2. Assert correct strain data
    expected_strain = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(strain, expected_strain)
    
    # 3. Assert correct time vector generation (4 seconds, 1Hz)
    expected_time = np.array([123456789.0, 123456790.0, 123456791.0, 123456792.0])
    assert np.allclose(time_vec, expected_time) 
    
    # 4. Assert correct channel dictionary creation 
    expected_mask = np.array([1, 1, 0, 0]) # Set by mock_read_hdf5_impl (3 -> 0b11)
    
    assert np.array_equal(channel_dict['DATA'], expected_mask)
    assert np.array_equal(channel_dict['SCIENCE'], expected_mask)
    assert np.array_equal(channel_dict['HW_INJ'], expected_mask)
    
    # DEFAULT channel is set to DATA
    assert np.array_equal(channel_dict['DEFAULT'], channel_dict['DATA'])
    
    # 5. Assert meta dictionary if tvec=False
    _, meta_dict, _ = loaddata('test.hdf5', ifo='H1', tvec=False)
    assert meta_dict['start'] == 123456789.0
    assert meta_dict['stop'] == 123456793.0
    assert meta_dict['dt'] == 1.0


# Test 2: SegmentList initialization
# We patch 'numpy.loadtxt' because 'SegmentList' calls it when reading from a file path.
@patch('numpy.loadtxt')
def test_segmentlist_init(mock_loadtxt):
    """
    Test SegmentList initialization from a list of segments and from a mock file 
    read using numpy.loadtxt.
    """
    
    # --- Test Case A: Initialization from a list ---
    input_list = [(100, 200), (300, 400)]
    seglist_a = SegmentList(input_list) 
    
    # 1. Assert the stored segment list is correct
    assert seglist_a.seglist == input_list
    
    # 2. Assert iteration and indexing work
    assert list(seglist_a) == input_list
    assert seglist_a[0] == (100, 200)

    # --- Test Case B: Initialization from a file (mocking np.loadtxt for 3-columns) ---
    mock_loadtxt.return_value = (
        np.array([500, 700]),  # start
        np.array([600, 800]),  # stop
        np.array([100, 100])   # duration
    )
    
    seglist_b = SegmentList('H1_segs.txt', numcolumns=3)
    
    # 3. Assert np.loadtxt was called with correct args
    mock_loadtxt.assert_called_with('H1_segs.txt', dtype='int', unpack=True)
    
    # 4. Assert the resulting segment list from the mock file data
    expected_seglist_b = [(500, 600), (700, 800)]
    assert seglist_b.seglist == expected_seglist_b

    # --- Test Case C: Initialization from a file with a single segment (scalar return) ---
    # The SegmentList handles scalar return from loadtxt for single-line files
    mock_loadtxt.return_value = (
        900,  # start (scalar)
        1000, # stop (scalar)
        100   # duration (scalar)
    )
    
    seglist_c = SegmentList('H1_single_seg.txt', numcolumns=3)
    
    # 5. Assert the correct segment list when np.loadtxt returns a scalar
    expected_seglist_c = [[900, 1000]]
    assert seglist_c.seglist == expected_seglist_c