import React, { createContext, useContext, useEffect, useState } from 'react';
import auth, { FirebaseAuthTypes } from '@react-native-firebase/auth';

type User = {
  uid: string;
  email: string | null;
  displayName: string | null;
};

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
  deleteAccount: (password: string) => Promise<void>;
  error: string | null;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

function mapFirebaseUser(firebaseUser: FirebaseAuthTypes.User | null): User | null {
  if (!firebaseUser) return null;
  return {
    uid: firebaseUser.uid,
    email: firebaseUser.email,
    displayName: firebaseUser.displayName,
  };
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Subscribe to auth state changes
    const unsubscribe = auth().onAuthStateChanged((firebaseUser) => {
      setUser(mapFirebaseUser(firebaseUser));
      setIsLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const handleSignUp = async (email: string, password: string, displayName: string) => {
    // Validate inputs
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

    const nameError = validateDisplayName(displayName);
    if (nameError) {
      setError(nameError);
      throw new Error(nameError);
    }

    setError(null);

    try {
      // Create user account
      const userCredential = await auth().createUserWithEmailAndPassword(
        email.trim(),
        password
      );

      // Update display name
      await userCredential.user.updateProfile({
        displayName: sanitizeDisplayName(displayName),
      });

      // Refresh user to get updated profile
      await auth().currentUser?.reload();
      setUser(mapFirebaseUser(auth().currentUser));
    } catch (err: any) {
      const message = getErrorMessage(err.code);
      setError(message);
      throw err;
    }
  };

  const handleSignIn = async (email: string, password: string) => {
    // Validate inputs
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

    setError(null);

    try {
      await auth().signInWithEmailAndPassword(email.trim(), password);
    } catch (err: any) {
      const message = getErrorMessage(err.code);
      setError(message);
      throw err;
    }
  };

  const logout = async () => {
    try {
      await auth().signOut();
    } catch (err: any) {
      const message = getErrorMessage(err.code);
      setError(message);
      throw err;
    }
  };

  const handleResetPassword = async (email: string) => {
    const emailError = validateEmail(email);
    if (emailError) {
      setError(emailError);
      throw new Error(emailError);
    }

    setError(null);

    try {
      await auth().sendPasswordResetEmail(email.trim());
    } catch (err: any) {
      const message = getErrorMessage(err.code);
      setError(message);
      throw err;
    }
  };

  const handleDeleteAccount = async (password: string) => {
    const currentUser = auth().currentUser;
    if (!currentUser || !currentUser.email) {
      const message = 'No user is currently signed in.';
      setError(message);
      throw new Error(message);
    }

    setError(null);

    try {
      // Re-authenticate user before deletion (required by Firebase for sensitive operations)
      const credential = auth.EmailAuthProvider.credential(
        currentUser.email,
        password
      );
      await currentUser.reauthenticateWithCredential(credential);

      // Delete the user account
      await currentUser.delete();
    } catch (err: any) {
      const message = getErrorMessage(err.code);
      setError(message);
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
        deleteAccount: handleDeleteAccount,
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
