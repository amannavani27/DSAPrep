import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  sendPasswordResetEmail,
  updateProfile,
  FirebaseAuthTypes,
} from '@react-native-firebase/auth';
import { doc, setDoc } from '@react-native-firebase/firestore';

type User = FirebaseAuthTypes.User;
import { auth, db } from '../config/firebase';

// Input validation utilities
function validateEmail(email: string): string | null {
  const trimmed = email.trim();
  if (!trimmed) return 'Email is required.';
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(trimmed)) return 'Please enter a valid email address.';
  return null;
}

function validatePassword(password: string): string | null {
  if (!password) return 'Password is required.';
  if (password.length < 6) return 'Password must be at least 6 characters.';
  return null;
}

function validateDisplayName(name: string): string | null {
  const trimmed = name.trim();
  if (!trimmed) return 'Display name is required.';
  if (trimmed.length < 2) return 'Display name must be at least 2 characters.';
  if (trimmed.length > 50) return 'Display name must be less than 50 characters.';
  if (!/^[a-zA-Z0-9\s\-_]+$/.test(trimmed)) {
    return 'Display name can only contain letters, numbers, spaces, hyphens, and underscores.';
  }
  return null;
}

function sanitizeDisplayName(name: string): string {
  return name.trim().slice(0, 50);
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  signUp: (email: string, password: string, displayName: string) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  resetPassword: (email: string) => Promise<void>;
  error: string | null;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
      setIsLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const handleSignUp = async (email: string, password: string, displayName: string) => {
    try {
      setError(null);

      // Validate inputs before making any API calls
      const emailError = validateEmail(email);
      if (emailError) {
        setError(emailError);
        throw new Error(emailError);
      }

      const passwordError = validatePassword(password);
      if (passwordError) {
        setError(passwordError);
        throw new Error(passwordError);
      }

      const displayNameError = validateDisplayName(displayName);
      if (displayNameError) {
        setError(displayNameError);
        throw new Error(displayNameError);
      }

      const sanitizedName = sanitizeDisplayName(displayName);
      const trimmedEmail = email.trim().toLowerCase();

      setIsLoading(true);
      const userCredential = await createUserWithEmailAndPassword(auth, trimmedEmail, password);

      // Update display name
      await updateProfile(userCredential.user, { displayName: sanitizedName });

      // Create user document in Firestore
      const userDocRef = doc(db, 'users', userCredential.user.uid);
      await setDoc(userDocRef, {
        email: trimmedEmail,
        displayName: sanitizedName,
        createdAt: new Date().toISOString(),
        progress: {},
        bookmarks: [],
      });
    } catch (err: any) {
      if (err.code) {
        setError(getErrorMessage(err.code));
      }
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignIn = async (email: string, password: string) => {
    try {
      setError(null);

      // Validate inputs before making any API calls
      const emailError = validateEmail(email);
      if (emailError) {
        setError(emailError);
        throw new Error(emailError);
      }

      if (!password) {
        setError('Password is required.');
        throw new Error('Password is required.');
      }

      const trimmedEmail = email.trim().toLowerCase();

      setIsLoading(true);
      await signInWithEmailAndPassword(auth, trimmedEmail, password);
    } catch (err: any) {
      if (err.code) {
        setError(getErrorMessage(err.code));
      }
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    try {
      setError(null);
      await signOut(auth);
    } catch (err: any) {
      setError(getErrorMessage(err.code));
      throw err;
    }
  };

  const handleResetPassword = async (email: string) => {
    try {
      setError(null);
      await sendPasswordResetEmail(auth, email);
    } catch (err: any) {
      setError(getErrorMessage(err.code));
      throw err;
    }
  };

  const clearError = () => setError(null);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        signUp: handleSignUp,
        signIn: handleSignIn,
        logout,
        resetPassword: handleResetPassword,
        error,
        clearError,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

function getErrorMessage(errorCode: string): string {
  switch (errorCode) {
    case 'auth/email-already-in-use':
      return 'This email is already registered. Please sign in instead.';
    case 'auth/invalid-email':
      return 'Please enter a valid email address.';
    case 'auth/operation-not-allowed':
      return 'Email/password accounts are not enabled.';
    case 'auth/weak-password':
      return 'Password should be at least 6 characters.';
    case 'auth/user-disabled':
      return 'This account has been disabled.';
    case 'auth/user-not-found':
      return 'No account found with this email.';
    case 'auth/wrong-password':
      return 'Incorrect password. Please try again.';
    case 'auth/invalid-credential':
      return 'Invalid email or password. Please try again.';
    case 'auth/too-many-requests':
      return 'Too many attempts. Please try again later.';
    default:
      return 'An error occurred. Please try again.';
  }
}
