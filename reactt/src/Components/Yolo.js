import React, { useState } from 'react';
import NavBar from './Navbar';
import { BsDownload } from 'react-icons/bs';
import { useEffect } from 'react';

const Yolo = () => {
  const [file, setFile] = useState(null);
  const [previewImage, setPreviewImage] = useState('');
  const [processedImage, setProcessedImage] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setProcessedImage(''); // Clear the processed image when a new file is selected

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewImage(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleSubmit = async () => {
    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('/yolo_crop', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        console.error('Response status:', response.status);
        throw new Error('Image processing failed');
      }

      const responseData = await response.json();

      if (responseData === 'None') {
        // Clear processed image and display error message in a pop-up
        setProcessedImage('');
        window.alert('Please try with other images or angles.');
      } else {
        const base64ImageData = responseData.base64Data;
        setProcessedImage(base64ImageData);
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      // Clear processed image and display error message in a pop-up
      setProcessedImage('');
      window.alert('Error processing image. Please try again.');
    }
  };

  const [downloadLink, setDownloadLink] = useState(''); // Add state to hold download link

  const handleDownload = () => {
    if (processedImage) {
      const link = document.createElement('a');
      link.href = `data:image/jpeg;base64,${processedImage}`;
      link.download = 'processed_image.jpg';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      window.alert('No processed image available to download');
    }
  };

  useEffect(() => {
    setDownloadLink(`data:image/jpeg;base64,${processedImage}`);
  }, [processedImage]);

  return (
    <div style={{ backgroundColor: 'lightcyan', minHeight: '100vh' }}>
      <NavBar />
      <div style={{ textAlign: 'center' }} className='mt-3'>
        <h2>Get Cropped Muzzle Pictures By Uploading Face Pictures</h2>
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <button onClick={handleSubmit}>Upload and Process Image</button>

        <div className='row mt-5'>

          <div className='col'>
            {previewImage && (
              <div>
                <h3>Input Image</h3>
                <img src={previewImage} alt="Preview" style={{ maxWidth: '300px' }} />
              </div>
            )}
          </div>

          <div className='col'>
            {processedImage && (
              <div className='row'>
                <div>
                  <h3>Processed Image</h3>
                  <img
                    src={`data:image/jpeg;base64,${processedImage}`}
                    alt="Processed"
                    style={{ maxWidth: '300px' }}
                    onError={(e) => {
                      console.error('Error loading processed image:', e);
                    }}
                  />
                </div>
                <div className='mt-4'>
                  <button onClick={handleDownload}>
                    Download <BsDownload />
                  </button>
                </div>
              </div>

            )}
          </div>

        </div>

      </div>
    </div>
  );
};

export default Yolo;
