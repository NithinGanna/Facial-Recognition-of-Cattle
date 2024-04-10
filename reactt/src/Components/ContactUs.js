import React from 'react'
import NavBar from './Navbar'

const ContactUs = () => {
  return (
    <div style={{ backgroundColor: 'lightcyan', minHeight: '100vh' }}>
    <NavBar/>
    <div className="container">
      <div className="row">
        <div className="col-lg-8 mx-auto text-center">
          <h1 className="mt-5 mb-4">Contact Us</h1>
          <p className="lead">
            We'd love to hear from you! Whether you have questions, feedback, or inquiries, our team is ready to assist you.
            Please feel free to reach out to us using the form below or through our contact details.
          </p>
          <p className="lead">
            Contact us via email at <a href="mailto:contact@CattleRecognitionproject.com">contact@cattledetectionproject.com</a> or
            call us at <a href="tel:+1234567890">+1 (234) 567-890</a>.
          </p>
          <p className="lead">
            Our office address:<br />
            1234 Cattle Drive,<br />
            Rural Town, Country
          </p>
        </div>
      </div>
    </div>
    </div>
  )
}

export default ContactUs

