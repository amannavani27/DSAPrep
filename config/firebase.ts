import auth from '@react-native-firebase/auth';
import firestore from '@react-native-firebase/firestore';

// React Native Firebase reads config from GoogleService-Info.plist (iOS)
// and google-services.json (Android) automatically
// No need to manually initialize - it's done by the native modules

// Export the module functions for convenience
// Usage: auth().signInWithEmailAndPassword(...) or firestore().collection(...)
export { auth, firestore };
