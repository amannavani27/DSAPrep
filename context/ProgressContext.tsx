import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import firestore from '@react-native-firebase/firestore';
import { TopicProgress, AppState, TopicType, Topic } from '../types';
import { dsaTopics } from '../data/topics';
import { systemDesignTopics } from '../data/systemDesignTopics';
import { useAuth } from './AuthContext';

const STORAGE_KEY = '@dsa_prep_state';

interface Stats {
  known: number;
  review: number;
  unseen: number;
  total: number;
  percentComplete: number;
}

interface FirestoreData {
  progress: Record<string, TopicProgress>;
  bookmarks: string[];
  selectedTopicType: TopicType;
  lastUpdated: number;
}

interface ProgressContextType {
  progress: Record<string, TopicProgress>;
  bookmarks: string[];
  selectedTopicType: TopicType;
  setSelectedTopicType: (type: TopicType) => void;
  markAsKnown: (topicId: string) => void;
  markForReview: (topicId: string) => void;
  toggleBookmark: (topicId: string) => void;
  isBookmarked: (topicId: string) => boolean;
  getTopicStatus: (topicId: string) => 'known' | 'review' | 'unseen';
  resetProgress: (topicType?: TopicType) => void;
  stats: Stats;
  dsaStats: Stats;
  systemDesignStats: Stats;
  getTopicsForType: (type: TopicType) => Topic[];
  allTopics: Topic[];
  isSyncing: boolean;
  syncError: string | null;
  hasPendingChanges: boolean;
}

const ProgressContext = createContext<ProgressContextType | undefined>(undefined);

const allTopics: Topic[] = [...dsaTopics, ...systemDesignTopics];

function calculateStats(
  topics: Topic[],
  progress: Record<string, TopicProgress>
): Stats {
  const total = topics.length;
  const known = topics.filter((t) => progress[t.id]?.status === 'known').length;
  const review = topics.filter((t) => progress[t.id]?.status === 'review').length;
  const unseen = total - known - review;
  const percentComplete = total > 0 ? Math.round((known / total) * 100) : 0;

  return { known, review, unseen, total, percentComplete };
}

export function ProgressProvider({ children }: { children: React.ReactNode }) {
  const { user, isAuthenticated } = useAuth();
  const [progress, setProgress] = useState<Record<string, TopicProgress>>({});
  const [bookmarks, setBookmarks] = useState<string[]>([]);
  const [selectedTopicType, setSelectedTopicTypeState] = useState<TopicType>('dsa');
  const [isLoaded, setIsLoaded] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [hasPendingChanges, setHasPendingChanges] = useState(false);

  // Track if update came from Firestore to prevent infinite loop
  const isRemoteUpdate = useRef(false);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastLocalUpdate = useRef<number>(0);

  // Load from local storage on mount
  useEffect(() => {
    loadLocalState();
  }, []);

  // Set up Firestore listener when user is authenticated
  useEffect(() => {
    if (!isAuthenticated || !user) {
      return;
    }

    const userDocRef = firestore().collection('users').doc(user.uid);

    // Listen for real-time updates from Firestore
    const unsubscribe = userDocRef.onSnapshot(
      (doc) => {
        if (doc.exists) {
          const data = doc.data() as FirestoreData;

          // Only update if the remote data is newer than our last local update
          if (data.lastUpdated && data.lastUpdated > lastLocalUpdate.current) {
            isRemoteUpdate.current = true;

            if (data.progress) {
              setProgress(data.progress);
            }
            if (data.bookmarks) {
              setBookmarks(data.bookmarks);
            }
            if (data.selectedTopicType) {
              setSelectedTopicTypeState(data.selectedTopicType);
            }

            // Also update local storage with remote data
            saveToLocalStorage(data.progress, data.bookmarks, data.selectedTopicType);

            // Reset the flag after a short delay to allow state updates to complete
            setTimeout(() => {
              isRemoteUpdate.current = false;
            }, 100);
          }
        }
      },
      (error) => {
        console.error('Firestore listener error:', error);
        setSyncError('Failed to sync with cloud. Changes saved locally.');
      }
    );

    // Initial sync: merge local and remote data
    syncWithFirestore();

    return () => unsubscribe();
  }, [isAuthenticated, user]);

  // Save to Firestore whenever progress changes (debounced)
  useEffect(() => {
    if (!isLoaded || isRemoteUpdate.current || !isAuthenticated || !user) {
      return;
    }

    // Debounce saves to prevent excessive writes
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    setHasPendingChanges(true);

    saveTimeoutRef.current = setTimeout(() => {
      saveToLocalStorage(progress, bookmarks, selectedTopicType);
      saveToFirestore();
    }, 500);

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [progress, bookmarks, selectedTopicType, isLoaded, isAuthenticated, user]);

  // Save to local storage when not authenticated
  useEffect(() => {
    if (!isLoaded || isRemoteUpdate.current || isAuthenticated) {
      return;
    }

    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(() => {
      saveToLocalStorage(progress, bookmarks, selectedTopicType);
    }, 500);

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [progress, bookmarks, selectedTopicType, isLoaded, isAuthenticated]);

  const loadLocalState = async () => {
    try {
      const stored = await AsyncStorage.getItem(STORAGE_KEY);
      if (stored) {
        const state: AppState = JSON.parse(stored);
        setProgress(state.progress || {});
        setBookmarks(state.bookmarks || []);
        setSelectedTopicTypeState(state.selectedTopicType || 'dsa');
      }
    } catch (error) {
      console.error('Failed to load local state:', error);
    } finally {
      setIsLoaded(true);
    }
  };

  const saveToLocalStorage = async (
    progressData: Record<string, TopicProgress> = progress,
    bookmarksData: string[] = bookmarks,
    topicType: TopicType = selectedTopicType
  ) => {
    const state: AppState = {
      progress: progressData,
      bookmarks: bookmarksData,
      currentDeckIndex: 0,
      selectedTopicType: topicType,
    };

    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
      console.error('Failed to save local state:', error);
    }
  };

  const syncWithFirestore = async () => {
    if (!user) return;

    setIsSyncing(true);
    setSyncError(null);

    try {
      const userDocRef = firestore().collection('users').doc(user.uid);
      const doc = await userDocRef.get();

      if (doc.exists) {
        const remoteData = doc.data() as FirestoreData;

        // Merge strategy: use the most recent data for each topic
        const mergedProgress = { ...remoteData.progress };

        Object.keys(progress).forEach((topicId) => {
          const localTopic = progress[topicId];
          const remoteTopic = remoteData.progress?.[topicId];

          if (!remoteTopic || (localTopic.lastSeen || 0) > (remoteTopic.lastSeen || 0)) {
            mergedProgress[topicId] = localTopic;
          }
        });

        // Merge bookmarks (union of both)
        const mergedBookmarks = Array.from(
          new Set([...(remoteData.bookmarks || []), ...bookmarks])
        );

        // Update local state with merged data
        isRemoteUpdate.current = true;
        setProgress(mergedProgress);
        setBookmarks(mergedBookmarks);

        setTimeout(() => {
          isRemoteUpdate.current = false;
        }, 100);

        // Save merged data back to Firestore
        const now = Date.now();
        lastLocalUpdate.current = now;

        await userDocRef.set({
          progress: mergedProgress,
          bookmarks: mergedBookmarks,
          selectedTopicType,
          lastUpdated: now,
        });

        // Update local storage
        await saveToLocalStorage(mergedProgress, mergedBookmarks, selectedTopicType);
      } else {
        // No remote data, upload local data
        await saveToFirestore();
      }
    } catch (error) {
      console.error('Failed to sync with Firestore:', error);
      setSyncError('Failed to sync with cloud. Changes saved locally.');
    } finally {
      setIsSyncing(false);
      setHasPendingChanges(false);
    }
  };

  const saveToFirestore = async () => {
    if (!user) return;

    setIsSyncing(true);
    setSyncError(null);

    try {
      const now = Date.now();
      lastLocalUpdate.current = now;

      await firestore().collection('users').doc(user.uid).set({
        progress,
        bookmarks,
        selectedTopicType,
        lastUpdated: now,
      });

      setHasPendingChanges(false);
    } catch (error) {
      console.error('Failed to save to Firestore:', error);
      setSyncError('Failed to sync with cloud. Changes saved locally.');
    } finally {
      setIsSyncing(false);
    }
  };

  const setSelectedTopicType = useCallback((type: TopicType) => {
    setSelectedTopicTypeState(type);
  }, []);

  const markAsKnown = useCallback((topicId: string) => {
    setProgress((prev) => ({
      ...prev,
      [topicId]: {
        topicId,
        status: 'known',
        lastSeen: Date.now(),
        reviewCount: (prev[topicId]?.reviewCount || 0) + 1,
      },
    }));
  }, []);

  const markForReview = useCallback((topicId: string) => {
    setProgress((prev) => ({
      ...prev,
      [topicId]: {
        topicId,
        status: 'review',
        lastSeen: Date.now(),
        reviewCount: (prev[topicId]?.reviewCount || 0) + 1,
      },
    }));
  }, []);

  const toggleBookmark = useCallback((topicId: string) => {
    setBookmarks((prev) =>
      prev.includes(topicId)
        ? prev.filter((id) => id !== topicId)
        : [...prev, topicId]
    );
  }, []);

  const isBookmarked = useCallback(
    (topicId: string) => bookmarks.includes(topicId),
    [bookmarks]
  );

  const getTopicStatus = useCallback(
    (topicId: string): 'known' | 'review' | 'unseen' => {
      return progress[topicId]?.status || 'unseen';
    },
    [progress]
  );

  const resetProgress = useCallback((topicType?: TopicType) => {
    if (topicType) {
      const topicsToReset = topicType === 'dsa' ? dsaTopics : systemDesignTopics;
      const topicIds = new Set(topicsToReset.map((t) => t.id));

      setProgress((prev) => {
        const newProgress = { ...prev };
        topicIds.forEach((id) => {
          delete newProgress[id];
        });
        return newProgress;
      });

      setBookmarks((prev) => prev.filter((id) => !topicIds.has(id)));
    } else {
      setProgress({});
      setBookmarks([]);
    }
  }, []);

  const getTopicsForType = useCallback((type: TopicType): Topic[] => {
    return type === 'dsa' ? dsaTopics : systemDesignTopics;
  }, []);

  const currentTopics = selectedTopicType === 'dsa' ? dsaTopics : systemDesignTopics;
  const stats = React.useMemo(
    () => calculateStats(currentTopics, progress),
    [currentTopics, progress]
  );

  const dsaStats = React.useMemo(
    () => calculateStats(dsaTopics, progress),
    [progress]
  );

  const systemDesignStats = React.useMemo(
    () => calculateStats(systemDesignTopics, progress),
    [progress]
  );

  return (
    <ProgressContext.Provider
      value={{
        progress,
        bookmarks,
        selectedTopicType,
        setSelectedTopicType,
        markAsKnown,
        markForReview,
        toggleBookmark,
        isBookmarked,
        getTopicStatus,
        resetProgress,
        stats,
        dsaStats,
        systemDesignStats,
        getTopicsForType,
        allTopics,
        isSyncing,
        syncError,
        hasPendingChanges,
      }}
    >
      {children}
    </ProgressContext.Provider>
  );
}

export function useProgress() {
  const context = useContext(ProgressContext);
  if (!context) {
    throw new Error('useProgress must be used within a ProgressProvider');
  }
  return context;
}
