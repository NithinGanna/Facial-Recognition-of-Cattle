import React from 'react';
import NavBar from './Navbar';

const Guidance = () => {
  return (
    <div style={{ backgroundColor: 'lightcyan', minHeight: '100vh' }}>
      <NavBar />
      <div className="container-fluid">
        <div className="row">
          <div className="col-lg-10">
            <div style={{ marginBottom: '30px' }} className='mt-4 pl-2'>

              <h3>Introduction</h3>
              <p>
                Welcome to Cattle recognition project , an innovative cattle recognition system designed to streamline cattle management processes.
                This guidance documentation aims to provide you with essential information on using our system effectively.
              </p>
            </div>

            <div style={{ marginBottom: '30px' }}>
              <h3>Getting Started</h3>
              <p>
                To begin using our cattle recognition system, [provide steps or instructions here, e.g., sign up, log in, upload images].
              </p>
            </div>

            <div style={{ marginBottom: '30px' }}>
              <h3>Features</h3>
              <p>
                Our system offers several key features including:
              </p>
              <ul>
                <li>Real-time cattle identification and tracking</li>
                <li>Health monitoring and anomaly detection</li>
                <li>Data-driven insights for herd management</li>
                {/* Add more features here */}
              </ul>
            </div>

            <div style={{ marginBottom: '30px' }}>
              <h3>How to Use</h3>
              <p>
                Here are some steps to effectively use our system:
              </p>
              <ol>
                <li>[Step 1: Describe the step]</li>
                <li>[Step 2: Describe the step]</li>
                {/* Add usage steps here */}
              </ol>
            </div>

            <div style={{ marginBottom: '30px' }}>
              <h3>FAQs (Frequently Asked Questions)</h3>
              <p>
                <strong>Q: How accurate is the cattle recognition?</strong><br />
                A: Our system achieves high accuracy by leveraging advanced algorithms and machine learning models.
                {/* Add more FAQs and their answers */}
              </p>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};

export default Guidance;
