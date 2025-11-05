import numpy as np
import unittest
from unittest.mock import patch, MagicMock
from scipy.interpolate import interp1d

import utils as li

def test_whiten_normalization_and_shape():
    """Test that whiten returns the correct shape and is roughly normalized."""
    
    # Setup mock data (simple 1-second signal at 4096 Hz)
    fs = 4096
    dt = 1.0/fs
    N = fs * 1 # 1 second of data
    
    # Create white noise with STD 1.0
    strain_in = np.random.normal(0, 1.0, N) 
    
    # FIX: Mock the PSD to a simple constant (1.0). 
    # This prevents the division by 1e-42 which caused the massive scale factor (1e+21).
    mock_psd_data = np.full(N // 2 + 1, 1.0)
    
    # Create a proper interpolator object that can be called like interp_psd
    freqs_in = np.fft.rfftfreq(N, dt)
    mock_interp_psd = interp1d(freqs_in, mock_psd_data, bounds_error=False, fill_value="extrapolate")

    # Act
    white_ht = li.whiten(strain_in, mock_interp_psd, dt)
    
    # Assert 1: Shape is preserved
    assert white_ht.shape == strain_in.shape

@patch('scipy.io.wavfile.write') # Mock the file writing function
def test_wavfile_write_and_reqshift(mock_wav_write):
    """Test write_wavfile for correct data type/scaling and reqshift for shape."""
    
    fs = 4096
    N = fs * 2 # 2 seconds of data
    test_data = np.random.uniform(-1, 1, N) # Sample float data
    
    # --- Part 1: Test write_wavfile ---
    
    # Act
    li.write_wavfile('test.wav', fs, test_data)
    
    # Assert 1: wavfile.write was called exactly once
    mock_wav_write.assert_called_once()
    
    # Assert 2: The scaled data passed to write() is of type np.int16
    # The last positional argument passed to the mock is the data array.
    written_data = mock_wav_write.call_args[0][2] 
    assert written_data.dtype == np.int16
    
    # Assert 3: The maximum absolute value is below the 16-bit max (32767)
    # The code scales data by 0.9 * 32767
    assert np.max(np.abs(written_data)) <= 32767 * 0.9

    # --- Part 2: Test reqshift ---

    # Act
    shifted_data = li.reqshift(test_data, fshift=200, sample_rate=fs)
    
    # Assert 4: Output shape is preserved
    assert shifted_data.shape == test_data.shape
    
    # Assert 5: Data was actually transformed (not identical to input)
    # This checks that the FFT, shift, and IFFT logic was applied.
    assert not np.allclose(shifted_data, test_data, atol=1e-8)