import React from 'react';

const NavBar = () => {
  return (
    <div>
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
          <a className="navbar-brand" href="/" style={{fontFamily: 'Lato',color: '#00ff26', fontSize: '28px' }}>
            Cattle Recognition
          </a>
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarNav"
            aria-controls="navbarNav"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav ml-auto">
              <li className="nav-item">
                <a className="nav-link" href="/" style={{ fontSize: '18px' }}>
                  Home
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="/about" style={{ fontSize: '18px' }}>
                  About Us
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="/guidance" style={{ fontSize: '18px' }}>
                  Guidance
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="/contact" style={{ fontSize: '18px' }}>
                  Contact Us
                </a>
              </li>
            </ul>
          </div>
      </nav>
    </div>
  );
};

export default NavBar;
