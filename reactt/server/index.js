// ------ basic express app

// import express from "express";
// const app = express();
// const port=8000;

// app.listen(port, ()=>{
//     console.log(`Server running on ${port}`)
// }) 


// -------to connect with my mongodb

// import mongoose from 'mongoose';

// const connect = mongoose.connect('mongodb://127.0.0.1:27017/Display');

// connect
//     .then(() => {
//         console.log('Database connected successfully');
//     })
//     .catch((err) => {
//         console.error('Database connection error:', err);
//     });

// ------- to write import we add  "type": "module", in package.json

// -------- to verify whteher it really connected to our db

// import mongoose from 'mongoose';
// import express from 'express';

// const DB_URL = 'mongodb://127.0.0.1:27017/Display';

// // Connect to MongoDB
// mongoose.connect(DB_URL)
// .then(async () => {
//   console.log('Database connected successfully');

//   // Fetch a list of collections
//   const collections = await mongoose.connection.db.listCollections().toArray();

//   if (collections.length > 0) {
//     // Display collection names and document counts
//     for (const collection of collections) {
//       const coll = mongoose.connection.collection(collection.name);
//       const count = await coll.countDocuments();
//       console.log(`Collection: ${collection.name} - Documents: ${count}`);
//     }
//   } else {
//     console.log('No collections found');
//   }
// })
// .catch((err) => {
//   console.error('Database connection error:', err);
// });

import mongoose from 'mongoose';
import express from 'express';
import cors from 'cors';

const DB_URL = 'mongodb://127.0.0.1:27017/Display';
const port = 8000; // Define the port

mongoose.connect(DB_URL)
.then(() => {
  console.log('Database connected successfully');
})
.catch((err) => {
  console.error('Database connection error:', err);
});

const app = express();
// Enable CORS for all routes
app.use(cors());

// Define your Mongoose schema for the 'cows' collection
const cowSchema = new mongoose.Schema({
  cattle_id: String,
  breed: String,
  dob: String,
  owner: String ,
  image: String,
  ph_no: {
    type: mongoose.Schema.Types.Number, // Define as Number for Int32
    min: -2147483648, // Minimum value for Int32
    max: 2147483647, // Maximum value for Int32
  },
  email: String,
  address: String,

});

// Register the model associated with the 'cows' collection
const Cow = mongoose.model('cows', cowSchema); // 'cows' is the collection name

// Endpoint to fetch a single document by cattle_id
app.get('/api/cows/:cattle_id', async (req, res) => {
  try {
    const { cattle_id } = req.params;
    const cow = await Cow.findOne({ cattle_id: cattle_id }); // Updated to use correct field name

    if (!cow) {
      return res.status(404).json({ message: 'Cow not found' });
    }

    console.log(`cow ${cattle_id} info done`);
    console.log(cow);
    res.json(cow);
  } catch (error) {
    console.log("error");
    res.status(500).json({ message: 'Server error' });
  }
});

// Endpoint to handle form submission and add data to MongoDB
// app.post('/submit_form', async (req, res) => {
//   try {
//     const lastCow = await Cow.findOne({}, {}, { sort: { cattle_id: -1 } }); // Find the last document by ID

//     let newId = 1; // Default value if no documents exist
//     if (lastCow) {
//       newId = parseInt(lastCow.cattle_id) + 1; // Increment the last document's cattle_id
//     }

//     console.log("newId is ",newId);

//     const formData = req.body;
//     formData.cattle_id = newId.toString(); // Set the new cattle_id for the form data

//     const createdCow = await Cow.create(formData); // Add the form data to the 'cows' collection
//     console.log('Created cow:', createdCow);

//     res.status(201).json(createdCow); // Respond with the created document
//   } catch (error) {
//     console.error('Error:', error);
//     res.status(500).json({ error: 'Failed to store data in MongoDB' });
//   }
// });


app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
