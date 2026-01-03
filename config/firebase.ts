import { getAuth } from '@react-native-firebase/auth';
import { getFirestore } from '@react-native-firebase/firestore';

// React Native Firebase reads config from GoogleService-Info.plist (iOS)
// and google-services.json (Android) automatically
// No need to manually initialize - it's done by the native modules

// Get Auth and Firestore instances using modular API
export const auth = getAuth();
export const db = getFirestore();
