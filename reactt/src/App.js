import './App.css';
import React  from 'react'

import { Routes , Route , BrowserRouter } from 'react-router-dom'
import Home from './Pages/Home';
import AboutUs from './Components/AboutUs';
import ContactUs from './Components/ContactUs';
import Guidance from './Components/Guidance';
import Verification from './Components/Verification';
import Cofirmation from './Components/Cofirmation';
import NewRegistration from './Components/NewRegistration';
import Confimed from './Components/Confimed';
import Yolo from './Components/Yolo';

function App() {

  return (
    <div>
      <BrowserRouter>
        <Routes>
          <Route path='/' element={ <Home/> } />

          <Route path='/about' element={ <AboutUs/> } />
          <Route path='/contact' element={ <ContactUs/> } />
          <Route path='/guidance' element={ <Guidance/> } />

          <Route path='/yolo' element={<Yolo/>} />

          <Route path='/Confirmation' element={<Cofirmation/>} />
          <Route path='/Verification' element={<Verification/>} />
          <Route path='/NewRegistration' element={<NewRegistration/>} />

          <Route path='/confirmed' element={<Confimed/>} />         

        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
