import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useAuth } from '../context/AuthContext';
import { router } from 'expo-router';

type AuthMode = 'login' | 'signup' | 'forgot';

export default function AuthScreen() {
  const { signIn, signUp, resetPassword, error, clearError, isLoading } = useAuth();

  const [mode, setMode] = useState<AuthMode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [displayName, setDisplayName] = useState('');

  const handleSubmit = async () => {
    clearError();

    if (!email.trim()) {
      Alert.alert('Error', 'Please enter your email');
      return;
    }

    if (mode === 'forgot') {
      try {
        await resetPassword(email);
        Alert.alert('Success', 'Password reset email sent. Check your inbox.');
        setMode('login');
      } catch (err) {
        // Error handled by context
      }
      return;
    }

    if (!password) {
      Alert.alert('Error', 'Please enter your password');
      return;
    }

    if (mode === 'signup') {
      if (!displayName.trim()) {
        Alert.alert('Error', 'Please enter your name');
        return;
      }
      if (password !== confirmPassword) {
        Alert.alert('Error', 'Passwords do not match');
        return;
      }
      if (password.length < 6) {
        Alert.alert('Error', 'Password must be at least 6 characters');
        return;
      }

      try {
        await signUp(email, password, displayName);
        router.replace('/(tabs)');
      } catch (err) {
        // Error handled by context
      }
    } else {
      try {
        await signIn(email, password);
        router.replace('/(tabs)');
      } catch (err) {
        // Error handled by context
      }
    }
  };

  const switchMode = (newMode: AuthMode) => {
    clearError();
    setMode(newMode);
    setPassword('');
    setConfirmPassword('');
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.header}>
          <Text style={styles.logo}>{'</>'}</Text>
          <Text style={styles.title}>Interview Prep</Text>
          <Text style={styles.subtitle}>
            {mode === 'login'
              ? 'Welcome back! Sign in to continue'
              : mode === 'signup'
              ? 'Create an account to save your progress'
              : 'Reset your password'}
          </Text>
        </View>

        <View style={styles.form}>
          {mode === 'signup' && (
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Name</Text>
              <TextInput
                style={styles.input}
                placeholder="Your name"
                placeholderTextColor="#6c7086"
                value={displayName}
                onChangeText={setDisplayName}
                autoCapitalize="words"
              />
            </View>
          )}

          <View style={styles.inputContainer}>
            <Text style={styles.label}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="your@email.com"
              placeholderTextColor="#6c7086"
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>

          {mode !== 'forgot' && (
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Password</Text>
              <TextInput
                style={styles.input}
                placeholder="Your password"
                placeholderTextColor="#6c7086"
                value={password}
                onChangeText={setPassword}
                secureTextEntry
              />
            </View>
          )}

          {mode === 'signup' && (
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Confirm Password</Text>
              <TextInput
                style={styles.input}
                placeholder="Confirm your password"
                placeholderTextColor="#6c7086"
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                secureTextEntry
              />
            </View>
          )}

          {error && (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          <TouchableOpacity
            style={[styles.submitButton, isLoading && styles.submitButtonDisabled]}
            onPress={handleSubmit}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.submitButtonText}>
                {mode === 'login'
                  ? 'Sign In'
                  : mode === 'signup'
                  ? 'Create Account'
                  : 'Send Reset Link'}
              </Text>
            )}
          </TouchableOpacity>

          {mode === 'login' && (
            <TouchableOpacity
              style={styles.forgotButton}
              onPress={() => switchMode('forgot')}
            >
              <Text style={styles.forgotButtonText}>Forgot password?</Text>
            </TouchableOpacity>
          )}
        </View>

        <View style={styles.footer}>
          {mode === 'forgot' ? (
            <TouchableOpacity onPress={() => switchMode('login')}>
              <Text style={styles.footerText}>
                Remember your password?{' '}
                <Text style={styles.footerLink}>Sign In</Text>
              </Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              onPress={() => switchMode(mode === 'login' ? 'signup' : 'login')}
            >
              <Text style={styles.footerText}>
                {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
                <Text style={styles.footerLink}>
                  {mode === 'login' ? 'Sign Up' : 'Sign In'}
                </Text>
              </Text>
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
  },
  scrollContent: {
    flexGrow: 1,
    padding: 24,
    justifyContent: 'center',
  },
  header: {
    alignItems: 'center',
    marginBottom: 40,
  },
  logo: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#e94560',
    marginBottom: 16,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#6c7086',
    textAlign: 'center',
  },
  form: {
    marginBottom: 24,
  },
  inputContainer: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    color: '#a6adc8',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#16213e',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: '#fff',
    borderWidth: 1,
    borderColor: '#1a1a2e',
  },
  errorContainer: {
    backgroundColor: 'rgba(233, 69, 96, 0.1)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(233, 69, 96, 0.3)',
  },
  errorText: {
    color: '#e94560',
    fontSize: 14,
    textAlign: 'center',
  },
  submitButton: {
    backgroundColor: '#e94560',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  submitButtonDisabled: {
    opacity: 0.7,
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  forgotButton: {
    alignItems: 'center',
    marginTop: 16,
  },
  forgotButtonText: {
    color: '#6c7086',
    fontSize: 14,
  },
  footer: {
    alignItems: 'center',
  },
  footerText: {
    color: '#6c7086',
    fontSize: 14,
  },
  footerLink: {
    color: '#e94560',
    fontWeight: '600',
  },
});
