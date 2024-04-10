import React, { useState, useEffect } from 'react';
import NavBar from './Navbar';
import { useLocation } from 'react-router-dom';

const Confirmed = () => {
  const [cowData, setCowData] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);

  const location = useLocation();
  const { state } = location;

  // Ensure the prediction message is received
  const prediction = state?.response?.prediction || 'No prediction available';
  const cattle_id = prediction; // Replace with the specific cattle ID

  useEffect(() => {
    fetch(`http://localhost:8000/api/cows/${cattle_id}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then((data) => {
        console.log('Received Data:', data); // Log received data
        setCowData(data);

        // Assuming the server sends binary image data as base64
        const base64Image = data.image; // Replace 'data.image' with the correct image property from the response

        console.log('Base64 Image:', base64Image); // Log base64 image data

        // Create a data URL from the base64 image
        const dataURL = `data:image/jpeg;base64,${base64Image}`;
        setImageSrc(dataURL);
      })
      .catch((error) => {
        console.error('Fetch error:', error); // Log fetch errors
      });
  }, [cattle_id]);

  const imageUrl = URL.createObjectURL(state?.image);

  return (
    <div style={{ backgroundColor: 'lightcyan', minHeight: '100vh' }}>
      <NavBar />
      <div className='row mt-5'>
        <div className='col'>
          <figure style={{ maxWidth: '100%', textAlign: 'center' }}>
            <img src={imageUrl} alt="Input" style={{ maxWidth: '100%', height: '200px', objectFit: 'contain' }} />
            <figcaption>Input</figcaption>
          </figure>
        </div>
        <div className='col'>
          <figure style={{ maxWidth: '100%', textAlign: 'center' }}>
            <img src={imageSrc} alt="Cattle Image" style={{ maxWidth: '100%', height: '200px', objectFit: 'contain' }} /> {/* Display the cow image */}
            <figcaption>Cattle {prediction}</figcaption>
          </figure>
        </div>
      </div>
      {cowData ? (
        <div style={{ textAlign: 'center' }} className='confirmed-card'>
          <h2 style={{ color: 'red' }}>Cattle {prediction} Details</h2>
          <p><strong>Cattle ID</strong> : {cowData.cattle_id}</p>
          <p><strong>Date of Birth</strong>: {cowData.dob}</p>
          <p><strong>Breed</strong> : {cowData.breed}</p>
          <p><strong>Owner</strong> : {cowData.owner}</p>
          <p><strong>Phone-No</strong> : {cowData.ph_no}</p>
          <p><strong>Email</strong> : {cowData.email}</p>
          <p><strong>Address</strong> : {cowData.address}</p>
          {/* Display other cow details as needed */}
        </div>
      ) : (
        <div style={{ textAlign: 'center' }} className='confirmed-card'>
          <span className="spinner-border spinner-border-sm text-primary" role="status" aria-hidden="true"></span>
          <span className="sr-only">Loading...</span>
        </div>
      )}
    </div>
  );
};

export default Confirmed;
