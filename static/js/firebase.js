// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.0/firebase-app.js";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.13.0/firebase-auth.js";
import { getFirestore, collection, addDoc } from "https://www.gstatic.com/firebasejs/10.13.0/firebase-firestore.js";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCJ6wo3ZnBY-TAuh84OQgEYNPnQK_kP-mQ",
  authDomain: "water-management-91e4a.firebaseapp.com",
  databaseURL: "https://water-management-91e4a-default-rtdb.firebaseio.com",
  projectId: "water-management-91e4a",
  storageBucket: "water-management-91e4a.appspot.com",
  messagingSenderId: "507816697694",
  appId: "1:507816697694:web:ef4c75c258bd789c7239cf",
  measurementId: "G-17BPD5WXYY"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

export async function login() {
    const email = document.getElementById('farmer_id').value;
    const password = document.getElementById('password').value;

    try {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        window.location.href = '/dashboard'; // Redirect to dashboard
    } catch (error) {
        alert(error.message); // Display error message
    }
}

export async function signUp() {
    const email = document.getElementById('farmer_id').value;
    const password = document.getElementById('password').value;
    const farmerName = document.getElementById('farmer_name').value; // Additional field for farmer name

    try {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // Add farmer details to Firestore
        await addDoc(collection(db, 'farmers'), {
            uid: user.uid,
            email: email,
            name: farmerName
        });

        window.location.href = '/login'; // Redirect to login page
    } catch (error) {
        alert(error.message); // Display error message
    }
}
