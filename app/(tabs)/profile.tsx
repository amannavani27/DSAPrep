import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ScrollView,
  Modal,
  TextInput,
  ActivityIndicator,
} from 'react-native';
import { useAuth } from '../../context/AuthContext';
import { useProgress } from '../../context/ProgressContext';

export default function ProfileScreen() {
  const { user, logout, deleteAccount } = useAuth();
  const { dsaStats, systemDesignStats, isSyncing } = useProgress();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletePassword, setDeletePassword] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleLogout = () => {
    Alert.alert(
      'Sign Out',
      'Are you sure you want to sign out? Your progress is saved to your account.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Sign Out',
          style: 'destructive',
          onPress: async () => {
            try {
              await logout();
            } catch (error) {
              console.error('Logout error:', error);
            }
          },
        },
      ]
    );
  };

  const handleDeleteAccountPress = () => {
    Alert.alert(
      'Delete Account',
      'Are you sure you want to permanently delete your account? This action cannot be undone and all your data will be lost.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete Account',
          style: 'destructive',
          onPress: () => {
            setDeleteError(null);
            setDeletePassword('');
            setShowDeleteModal(true);
          },
        },
      ]
    );
  };

  const handleConfirmDelete = async () => {
    if (!deletePassword) {
      setDeleteError('Please enter your password.');
      return;
    }

    setIsDeleting(true);
    setDeleteError(null);

    try {
      await deleteAccount(deletePassword);
      setShowDeleteModal(false);
    } catch (error: any) {
      setDeleteError(error.message || 'Failed to delete account. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleCancelDelete = () => {
    setShowDeleteModal(false);
    setDeletePassword('');
    setDeleteError(null);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>
            {user?.displayName?.charAt(0).toUpperCase() || user?.email?.charAt(0).toUpperCase() || '?'}
          </Text>
        </View>
        <Text style={styles.name}>{user?.displayName || 'User'}</Text>
        <Text style={styles.email}>{user?.email}</Text>
        {isSyncing && (
          <Text style={styles.syncingText}>Syncing...</Text>
        )}
      </View>

      <View style={styles.statsSection}>
        <Text style={styles.sectionTitle}>Your Progress</Text>

        <View style={styles.statsCard}>
          <Text style={styles.statsCardTitle}>DSA</Text>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{dsaStats.known}</Text>
              <Text style={styles.statLabel}>Known</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{dsaStats.review}</Text>
              <Text style={styles.statLabel}>Review</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{dsaStats.unseen}</Text>
              <Text style={styles.statLabel}>Unseen</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={[styles.statValue, styles.percentValue]}>{dsaStats.percentComplete}%</Text>
              <Text style={styles.statLabel}>Complete</Text>
            </View>
          </View>
        </View>

        <View style={styles.statsCard}>
          <Text style={styles.statsCardTitle}>System Design</Text>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{systemDesignStats.known}</Text>
              <Text style={styles.statLabel}>Known</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{systemDesignStats.review}</Text>
              <Text style={styles.statLabel}>Review</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{systemDesignStats.unseen}</Text>
              <Text style={styles.statLabel}>Unseen</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={[styles.statValue, styles.percentValue]}>{systemDesignStats.percentComplete}%</Text>
              <Text style={styles.statLabel}>Complete</Text>
            </View>
          </View>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Account</Text>

        <TouchableOpacity style={styles.menuItem} onPress={handleLogout}>
          <Text style={styles.menuItemText}>Sign Out</Text>
          <Text style={styles.menuItemIcon}>{'>'}</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.menuItem, styles.deleteMenuItem]} onPress={handleDeleteAccountPress}>
          <Text style={styles.deleteMenuItemText}>Delete Account</Text>
          <Text style={styles.menuItemIcon}>{'>'}</Text>
        </TouchableOpacity>
      </View>

      <Modal
        visible={showDeleteModal}
        transparent
        animationType="fade"
        onRequestClose={handleCancelDelete}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Confirm Account Deletion</Text>
            <Text style={styles.modalDescription}>
              Please enter your password to confirm account deletion. This action is permanent and cannot be undone.
            </Text>

            <TextInput
              style={styles.passwordInput}
              placeholder="Enter your password"
              placeholderTextColor="#6c7086"
              secureTextEntry
              value={deletePassword}
              onChangeText={setDeletePassword}
              editable={!isDeleting}
              autoCapitalize="none"
            />

            {deleteError && (
              <Text style={styles.errorText}>{deleteError}</Text>
            )}

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.cancelButton]}
                onPress={handleCancelDelete}
                disabled={isDeleting}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.modalButton, styles.confirmDeleteButton]}
                onPress={handleConfirmDelete}
                disabled={isDeleting}
              >
                {isDeleting ? (
                  <ActivityIndicator color="#fff" size="small" />
                ) : (
                  <Text style={styles.confirmDeleteButtonText}>Delete</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      <View style={styles.footer}>
        <Text style={styles.footerText}>Interview Prep v1.0</Text>
        <Text style={styles.footerSubtext}>Your progress is automatically synced</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
  },
  content: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    paddingVertical: 30,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#e94560',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  avatarText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
  },
  name: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  email: {
    fontSize: 14,
    color: '#6c7086',
  },
  syncingText: {
    fontSize: 12,
    color: '#e94560',
    marginTop: 8,
  },
  statsSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 16,
  },
  statsCard: {
    backgroundColor: '#16213e',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
  },
  statsCardTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#a6adc8',
    marginBottom: 12,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  percentValue: {
    color: '#e94560',
  },
  statLabel: {
    fontSize: 12,
    color: '#6c7086',
  },
  section: {
    marginBottom: 24,
  },
  menuItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#16213e',
    borderRadius: 12,
    padding: 16,
  },
  deleteMenuItem: {
    marginTop: 12,
    backgroundColor: 'rgba(220, 38, 38, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(220, 38, 38, 0.3)',
  },
  menuItemText: {
    fontSize: 16,
    color: '#e94560',
  },
  deleteMenuItemText: {
    fontSize: 16,
    color: '#dc2626',
  },
  menuItemIcon: {
    fontSize: 16,
    color: '#6c7086',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#16213e',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 12,
    textAlign: 'center',
  },
  modalDescription: {
    fontSize: 14,
    color: '#a6adc8',
    marginBottom: 20,
    textAlign: 'center',
    lineHeight: 20,
  },
  passwordInput: {
    backgroundColor: '#0f0f23',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: '#fff',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#313244',
  },
  errorText: {
    color: '#dc2626',
    fontSize: 14,
    marginBottom: 12,
    textAlign: 'center',
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  modalButton: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cancelButton: {
    backgroundColor: '#313244',
  },
  cancelButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  confirmDeleteButton: {
    backgroundColor: '#dc2626',
  },
  confirmDeleteButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  footer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  footerText: {
    fontSize: 14,
    color: '#6c7086',
    marginBottom: 4,
  },
  footerSubtext: {
    fontSize: 12,
    color: '#45475a',
  },
});
