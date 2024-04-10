import React , {useState} from 'react';
import { useNavigate } from 'react-router-dom';


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
              navigate('/NotVerify',{
                state: {image: fileInput.files[0]}
              });
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
        <div className="row justify-content-center">
          <div className="col-lg-6 col-md-8">
            <form
              id="imageForm"
              // action="/upload"
              method="post"
              encType="multipart/form-data"
              onSubmit={validateForm}
            >
              <div className="mb-3">
                <label htmlFor="imageUpload" className="form-label col">
                  Upload Image like
                </label>
                <img
                  src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSJWCJIyNqbsV1_G_Fv77Fm3cNRcL76E0EgjQ&usqp=CAU'
                  className='row pb-3 w-100'
                  style={{ maxHeight: '300px', objectFit: 'cover' }}
                />
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
  


