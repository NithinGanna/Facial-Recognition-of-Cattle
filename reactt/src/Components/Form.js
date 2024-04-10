import React , {useState} from 'react';
import { useNavigate } from 'react-router-dom';
import Modal from 'react-modal';
import { BsExclamationTriangleFill } from 'react-icons/bs';
import { AiOutlineCloseCircle } from 'react-icons/ai'

const ModalContent = ({ closeModal, message }) => {
  return (
    <div style={{ padding: '20px' }}>
      {/* <BsExclamationTriangleFill size={80} style={{ color: '#ff5959' }} className="bi flex-shrink-0 me-2 " /> */}
      <AiOutlineCloseCircle size={80} style={{ color: '#ff5959' }} className="bi flex-shrink-0 me-2" />
      <h1 style={{ color: '#ff5959' }}>{message}</h1>
      <p>Go and get Registered by clicking below</p>
      <h6><a href='/NewRegistration'>Wanna Register??</a></h6>
      <button
        onClick={closeModal}
        style={{
          backgroundColor: 'red' ,
          color: 'white',
          border: 'none',
          padding: '10px',
          borderRadius: '5px',
          cursor: 'pointer',
        }}
        className='btn w-25 mt-2'
      >
        Ok
      </button>
    </div>
  );
};


const Form = () => {

  const openFloatingWindow = () => {
    const fileInput = document.getElementById('imageUpload');

    if (fileInput.checkValidity()) {
      const file = fileInput.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
          const imageUrl = event.target.result;
          const displayedImage = document.getElementById('displayedImage');
          displayedImage.src = imageUrl;
          document.getElementById('floatingWindow').style.display = 'block';
        };
        reader.readAsDataURL(file);
      }
    } else {
      fileInput.classList.add('is-invalid');
    }
  };

  const closeFloatingWindow = () => {
    document.getElementById('floatingWindow').style.display = 'none';
  };

  const validateForm = () => {
    const selectedOption = document.getElementById('selectOption').value;
    if (selectedOption === '') {
      alert('Please select an option.');
      return false;
    }
    document.getElementById('selectedOption').value = selectedOption;
    return true;
  };

  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const [modalIsOpen, setModalIsOpen] = useState(false);

  const openModal = () => {
    setModalIsOpen(true);
  };

  const closeModal = () => {
    setModalIsOpen(false);
  };

  const [modalMessage, setModalMessage] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById('imageUpload');
    const selectedOption = document.getElementById('selectOption').value;

    if (!fileInput.files[0] || selectedOption === '') {
      alert('Please upload an image and select an option.');
      return;
    }

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    if (selectedOption === 'Verification') {
      setIsLoading(true); // Activate loading spinner

      try {
        const response = await fetch('/Verify', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const responseData = await response.json();
          console.log('Response from Flask:', responseData);
          console.log('Type of response data:', typeof responseData);

          const prediction = responseData.prediction; // Check the correct object key
          
          if (prediction === "cant find this registration") {
            openModal();
            setModalMessage('Not Registered');
          } else{
            navigate(`/Verification`, {
              state: { image: fileInput.files[0], response: responseData},
            });
          }
        } else {
          throw new Error('Failed to process image');
        }
      } catch (error) {
        console.error('Error:', error);
        // Handle error, show error message, etc.
      } finally {
        setIsLoading(false); // Deactivate loading spinner
      }
    }
    else if(selectedOption === 'Confirmation')
    {
      navigate('/Confirmation',{
        state: {image: fileInput.files[0]}
      });
    }
    else if(selectedOption === 'New Registration')
    {
      navigate('/NewRegistration',{
        state: {image: fileInput.files[0]}
      });
    }
  };

  const handleSelectOptionChange = (event) => {
    const selectedOption = event.target.value;
    switch (selectedOption) {
      case 'Verification':
        console.log('Verification selected');
        break;
      case 'Confirmation':
        console.log('Confirmation selected');
        break;
      case 'New Registration':
        console.log('New Registration selected');
        break;
      default:
        console.log('No option selected');
    }
  };


  return (
    <div>
      <div className="container">
        <Modal
          isOpen={modalIsOpen}
          onRequestClose={closeModal}
          contentLabel="Error Modal"
          style={modalStyles.modal}
        >
          <ModalContent closeModal={closeModal} message={modalMessage} />
        </Modal>
        <div className="row justify-content-center">
          <div className="col-lg-6 col-md-8">
            <form
              id="imageForm"
              // action="/upload"
              method="post"
              encType="multipart/form-data"
              onSubmit={validateForm}
              style={{backgroundColor: 'white'}}
            >
              <div className="mb-3">
                <label htmlFor="imageUpload" className="form-label col">
                  Upload Image
                </label>
                <input
                  type="file"
                  className="form-control"
                  id="imageUpload"
                  name="image"
                  accept="image/*"
                  required
                />
                <div className="invalid-feedback">Please upload an image.</div>
              </div>
              <div>
                <p>Get cropped Muzzle pictures out of Cattle Face <a href='/yolo'>from here</a> ?? </p>
              </div>
              <div className="row mb-3">
                <div className="col-5">
                  <button
                    type="button"
                    className="btn btn-outline-danger w-100"
                    onClick={openFloatingWindow}
                  >
                    View
                  </button>
                </div>
                <div className="col-7">
                  <select
                    id="selectOption"
                    className="form-select form-select-sm w-100"
                    aria-label="Small select example"
                    required
                    onChange={handleSelectOptionChange}
                  >
                    <option value="" selected disabled>
                      Choose an option
                    </option>
                    <option value="Verification">Verification</option>
                    <option value="Confirmation">Confirmation</option>
                    <option value="New Registration">New Registration</option>
                  </select>
                  {/* Hidden input to store selected option */}
                  <input type="hidden" id="selectedOption" name="selectedOption" />
                </div>
              </div>
              <div className="text-center">
                <button type="submit" className="btn btn-primary w-50" onClick={handleSubmit}>
                  Submit
                </button>
              </div>
              <div className='loader-container'>
                {isLoading && <div className="loader"></div>}
              </div>
            </form>
          </div>
        </div>
      </div>

      {/* Floating window for displaying the image */}
      <div id="floatingWindow" className="floating-window">
        <span className="close-btn" onClick={closeFloatingWindow}>
          &times;
        </span>
        <div className="floating-content">
          <img id="displayedImage" src="" alt="Displayed Image" />
        </div>
      </div>
    </div>
  );
};

export default Form;



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




