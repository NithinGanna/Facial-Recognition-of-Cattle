import React , { useState, useEffect }  from 'react'
import NavBar from './Navbar'
import { BsExclamationTriangleFill } from 'react-icons/bs';
import { AiOutlineCloseCircle } from 'react-icons/ai';
import { useLocation } from 'react-router-dom';
import Modal from 'react-modal';
import { FaSyncAlt } from 'react-icons/fa';

const ModalContent = ({ closeModal, message }) => {
  return (
    <div style={{ padding: '20px' }}>
      {/* <BsExclamationTriangleFill size={80} style={{ color: '#ff5959' }} className="bi flex-shrink-0 me-2 " /> */}
      <AiOutlineCloseCircle size={80} style={{ color: '#ff5959' }} className="bi flex-shrink-0 me-2" />
      <h2 style={{ color: '#ff5959' }}>{message}</h2>
      <p>Cannot be Registered</p>
      <p>Fake entrances are not Allowed/Encouraged</p>
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

const AnotherModalContent = ({ closeModal }) => {
  // This modal will appear when you want to upload multiple images
  // Implement the UI for this modal window as per your requirements
    const [images, setImages] = useState([]);
    const [uploadedCount, setUploadedCount] = useState(0);
    const [errorMessage, setErrorMessage] = useState('');

  const handleImageChangeOfNew = (event) => {
    const selectedFiles = event.target.files;
    const filesArray = Array.from(selectedFiles).slice(0, 10 - uploadedCount); // Limit to remaining slots
    setImages([...images, ...filesArray]);
    setUploadedCount(uploadedCount + filesArray.length);
  };

  const [uploading, setUploading] = useState(false);

  const handleSubmitOfNew = async (event) => {
    event.preventDefault();
    // Check if minimum 5 images are uploaded
    if (uploadedCount >= 5) {
      setUploading(true); // Activate the loader
      const formData = new FormData();
      images.forEach((image, index) => {
        formData.append(`image${index}`, image);
      });

      try {
        const response = await fetch('/upload_images', {
          method: 'POST',
          body: formData,
        });


        if (response.ok) {

          const responseData = await response.json();
          console.log('Response from Flask:', responseData);
          console.log('Type of response data:', typeof responseData);

          // Handle success - images uploaded to the backend
          closeModal(); // Close the modal or perform any necessary actions

          newId = responseData.progress;
          alert(`Registration was done with ${newId}`);
          return;

        } else {
          // Handle error in uploading images
          console.error('Failed to upload images');
        }
      } catch (error) {
        // Handle fetch error
        console.error('Error uploading images:', error);
      }finally {
        setUploading(false); // Deactivate the loader
      }
    } else {
      setErrorMessage('Upload at least 5 to 10 images');
    }
  };


  const handleRefresh = () => {
    setImages([]);
    setUploadedCount(0);
    setErrorMessage('');
  };

  return (
    <div style={{ padding: '10px' }}>
      <h2>Upload Cattle Muzzle Images Only</h2>
      <p>Upload aleast 5 to 10 images</p>
      <form onSubmit={handleSubmitOfNew}>
        <div className='row'>
          <div className='col-8'>
            <p>Uploaded : {uploadedCount} / 10</p>
          </div>
          <div className='col-1'>
            <button type="button" style={{backgroundColor:'silver'}} onClick={handleRefresh}>
              <FaSyncAlt /> {/* Icon for refresh */}
            </button>
          </div>
        </div>
        <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleImageChangeOfNew}
              required
        />
        <button style={{
          color: 'white',
          border: 'none',
          padding: '10px',
          borderRadius: '5px',
          cursor: 'pointer',
        }}
        className='btn btn-primary w-25 mt-2' type="submit">Upload</button>

      </form>
      {uploading && <div className="register-loader mt-0 mb-1"></div>}
      {errorMessage && <p style={{ color: 'red' }} className='mt-2'>{errorMessage}</p>}
    </div>
  );
};

const NewRegistration = () => {

  const [cattleData, setCattleData] = useState({
    owner: '',
    ph_no:'',
    email:'',
    address:'',
    breed: '',
    dob: '',
    image: '',
    // Add more fields as needed for cattle details
  });

  const handleChange = (event) => {
    const { name, value } = event.target;
    setCattleData({ ...cattleData, [name]: value });
  };

  // const handleImageChange = (event) => {
  //   const imageFile = event.target.files[0];
  //   setCattleData({ ...cattleData, image: imageFile });
  // };

  const handleImageChange = async (event) => {
    const imageFile = event.target.files[0];

    if (imageFile) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result; // Get the base64 representation of the image
        setCattleData({ ...cattleData, image: base64String });
      };
      reader.readAsDataURL(imageFile); // Read the file as a data URL (base64 format)
    }
  };
  
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadImagesModalOpen, setUploadImagesModalOpen] = useState(false);

  const openModal = () => {
    setModalIsOpen(true);
  };

  const closeModal = () => {
    setModalIsOpen(false);
  };

  const openUploadImagesModal = () => {
    setUploadImagesModalOpen(true);
  };

  const closeUploadImagesModal = () => {
    setUploadImagesModalOpen(false);
  };

  const location = useLocation();
  const { state } = location;

  const imageUrl = state?.response?.image || '';

  const [modalMessage, setModalMessage] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true); // Activate loading spinner
  
    try {

      const formData = new FormData();
      formData.append('image', state.image); // Accessing image from the state received via routing

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
          console.log("if loop",prediction);

          const response = await fetch('/submit_form', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(cattleData), // Sending the cattle data to be added to MongoDB
          });

          if (response.ok) {
            const responseData = await response.json();
            console.log('Response from backend:', responseData);
            // Handle success - data added to MongoDB
          } else {
            console.error('Failed to submit form data to MongoDB');
            // Handle submission failure
          }

          // openModal();
          // setModalMessage('Will be Registered');
          openUploadImagesModal(); // Opens another modal for uploading multiple images
        } else{
          console.log("else loop",prediction);
          openModal();
          setModalMessage('Already Registered');
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
    // Handle cattle registration logic here
    console.log('Cattle data:', cattleData);
    // Add your logic to submit cattle data to the server
  };

  return (
    <div style={{ backgroundColor: 'lightcyan', minHeight: '100vh' }}>
    <NavBar/>
    <div className="container mt-5">
      <Modal
        isOpen={modalIsOpen}
        onRequestClose={closeModal}
        contentLabel="Error Modal"
        style={modalStyles.modal}
      >
        <ModalContent closeModal={closeModal} message={modalMessage} />
      </Modal>
      <div className="row justify-content-center">
        <div className="col-md-6">
          <h2>New Registration</h2>
          <form onSubmit={handleSubmit}>
            <div className="mb-3">
              <label htmlFor="name" className="form-label">Your Name (Owner)</label>
              <input type="text" className="form-control" id="owner" name="owner" value={cattleData.owner}
          onChange={handleChange} required />
            </div>
            <div className="mb-3 row">
              <div className='col'>
                <label htmlFor="ph_no" className="form-label">Phone Number</label>
                <input type="tel" className="form-control" id="ph_no" name="ph_no" value={cattleData.ph_no}
            onChange={handleChange} required />
              </div>
              <div className='col'>
                <label htmlFor="email" className="form-label">Email</label>
                <input type="email" className="form-control" id="email" name="email" value={cattleData.email}
            onChange={handleChange} required />
              </div>
            </div>
            <div className="mb-3">
              <label htmlFor="address" className="form-label">Address</label>
              <input type="text" className="form-control" id="address" name="address" value={cattleData.address}
          onChange={handleChange} required />
            </div>
            <div className="mb-3 row">
              <div className='col'>
                <label htmlFor="breed" className="form-label">Breed</label>
                <input type="text" className="form-control" id="breed" name="breed" value={cattleData.breed}
            onChange={handleChange} required />
              </div>
              <div className='col'>
                <label htmlFor="dob" className="form-label">Date of Birth</label>
                <input type="date" className="form-control" id="dob" name="dob" value={cattleData.dob}
            onChange={handleChange} required />
              </div>
            </div>
            <div className="mb-4">
              <label htmlFor="image" className="form-label">Upload Face Image</label>
              <input type="file" className="form-control" id="image" name="image" onChange={handleImageChange}
              required/>
            </div>
            <button type="submit" className="btn btn-primary" >Submit</button>
            <div className='loader-container'>
              {isLoading && <div className="newR-loader"></div>}
            </div>
          </form>
          <Modal
            isOpen={uploadImagesModalOpen}
            onRequestClose={closeUploadImagesModal}
            contentLabel="Upload Multiple Images Modal"
            style={modalStyles.modal}
          >
            <AnotherModalContent closeModal={closeUploadImagesModal} />
          </Modal>
        </div>
      </div>
    </div>
    </div>
  );
}

export default NewRegistration

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




