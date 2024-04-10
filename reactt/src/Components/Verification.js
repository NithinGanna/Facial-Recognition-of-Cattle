import React , { useState, useEffect } from 'react';
import { Navigate, redirect, useLocation, useNavigate } from 'react-router-dom';
import NavBar from './Navbar';
import axios from 'axios'; 

const Verification = () => {
  const location = useLocation();
  const { state } = location;
  const navigate = useNavigate();

  // Ensure the prediction message is received
  const prediction = state?.response?.prediction || 'No prediction available';
  // Assuming 'image' is a URL string
  // const imageUrl = state?.response?.image || '';
  const imageUrl = URL.createObjectURL(state?.image);
  
  const [imageSrc,setImageSrc] = useState(null);

  const [cowData, setCowData] = useState(null);

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

  return (
    <div style={{ backgroundColor: 'lightcyan', minHeight: '100vh' }}>
      <NavBar/>

      <div className='container'>
        <h1 style={{color: 'green',textAlign: 'center'}} className='verified-line'>Verified!!</h1>

        <div className='row'>
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
          {/* Display other cow details as needed */}
        </div>
      ) : (
        // <p style={{textAlign: 'center'}}>Waiting for MongoDB...</p>
        <div style={{ textAlign: 'center' }} className='confirmed-card'>
          <span class="spinner-border spinner-border-sm text-primary" role="status" aria-hidden="true"></span>
          <span class="sr-only">Loading...</span>
        </div>
      )}
      
    </div>
  );
};

export default Verification;




