
import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Modal from 'react-modal';
import NavBar from './Navbar';
import { BsExclamationTriangleFill } from 'react-icons/bs';
import { AiOutlineCloseCircle } from 'react-icons/ai';

const ModalContent = ({ closeModal, message }) => {
  return (
    <div style={{ padding: '20px' }}>
      {/* <BsExclamationTriangleFill size={80} style={{ color: '#ff5959' }} className="bi flex-shrink-0 me-2 " /> */}
      <AiOutlineCloseCircle size={80} style={{ color: '#ff5959' }} className="bi flex-shrink-0 me-2" />
      <h2 style={{ color: '#ff5959' }}>{message}</h2>
      <p>Please try again or go to Verification or register</p>
      <button
        onClick={closeModal}
        style={{
          backgroundColor: '#ff5959',
          color: 'white',
          border: 'none',
          padding: '10px',
          borderRadius: '5px',
          cursor: 'pointer',
        }}
        className='btn w-25'
      >
        Ok
      </button>
    </div>
  );
};

const Confirmation = () => {
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { state } = location;

  const openModal = () => {
    setModalIsOpen(true);
  };

  const closeModal = () => {
    setModalIsOpen(false);
  };

  const [modalMessage, setModalMessage] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();

    // Get the image data
    const imageData = state?.image;

    const cattleID = document.getElementById('numberInput').value;

    const formData = new FormData();
    formData.append('image', imageData);
    formData.append('cattleID', cattleID);

    try {
      const response = await fetch('/confirm', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const responseData = await response.json();

        const prediction = responseData.prediction;

        if (prediction === 'Not Matched') {
          openModal();
          setModalMessage('The entered ID does not match.');
          // navigate('/');
        } else if (prediction === 'id does not exist') {
          openModal();
          setModalMessage('The entered ID does not exist.');
        } else {
          navigate('/confirmed', { state: { image: imageData, response: responseData } });
        }
      } else {
        console.error('Failed to upload image');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div style={{ backgroundColor: 'lightcyan', minHeight: '100vh' }}>
      <NavBar/>
      <Modal
        isOpen={modalIsOpen}
        onRequestClose={closeModal}
        contentLabel="Error Modal"
        style={modalStyles.modal}
      >
        <ModalContent closeModal={closeModal} message={modalMessage} />
      </Modal>
      <div>
        <div style={{ textAlign: 'center' }} className="mt-5">
          {state?.image && <img src={URL.createObjectURL(state.image)} id="img-id" alt="Uploaded" />}
          <h5>Input</h5>
        </div>
        <div className="container mt-5">
          <div className="row justify-content-center">
            <div className="col-md-6">
              <div className="card">
                <div className="card-body">
                  <h2 className="card-title text-center mb-4">Confirmation</h2>
                  <form action="/home" method="post" style={{ backgroundColor: 'white' }}>
                    <div className="mb-3">
                      <label htmlFor="numberInput" className="form-label">Enter your Cattle ID</label>
                      <input type="number" className="form-control" id="numberInput" name="numberInput" required />
                    </div>
                    <div className="text-center">
                      <button type="submit" className="btn btn-primary" onClick={handleSubmit}>Submit</button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Confirmation;

// Modal styles
const modalStyles = {
  modal: {
    content: {
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      backgroundColor: 'white',
      padding: '20px',
      borderRadius: '8px',
      width: '60%',
      maxWidth: '400px',
      textAlign: 'center',
      boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
    },
    overlay: {
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
    },
  },
};

