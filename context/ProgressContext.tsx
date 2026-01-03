import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { doc, getDoc, setDoc, onSnapshot } from '@react-native-firebase/firestore';
import { TopicProgress, AppState, TopicType, Topic } from '../types';
import { dsaTopics } from '../data/topics';
import { systemDesignTopics } from '../data/systemDesignTopics';
import { db } from '../config/firebase';
import { useAuth } from './AuthContext';

const STORAGE_KEY = '@dsa_prep_state';

interface Stats {
  known: number;
  review: number;
  unseen: number;
  total: number;
  percentComplete: number;
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

  // Load from local storage on mount
  useEffect(() => {
    loadLocalState();
  }, []);

  // Sync with Firestore when user logs in
  useEffect(() => {
    if (!isAuthenticated || !user) {
      return;
    }

    const userDocRef = doc(db, 'users', user.uid);

    // Listen for real-time updates from Firestore
    const unsubscribe = onSnapshot(
      userDocRef,
      (docSnapshot) => {
        if (docSnapshot.exists()) {
          const data = docSnapshot.data();
          // Mark as remote update to prevent saving back to Firestore
          isRemoteUpdate.current = true;

          if (data?.progress) {
            setProgress(data.progress);
          }
          if (data?.bookmarks) {
            setBookmarks(data.bookmarks);
          }
          if (data?.selectedTopicType) {
            setSelectedTopicTypeState(data.selectedTopicType);
          }

          // Reset flag after state updates are processed
          setTimeout(() => {
            isRemoteUpdate.current = false;
          }, 100);
        }
      },
      (error) => {
        console.error('Firestore snapshot error:', error);
      }
    );

    return () => unsubscribe();
  }, [isAuthenticated, user]);

  // Save to local storage whenever progress changes (debounced)
  useEffect(() => {
    if (!isLoaded || isRemoteUpdate.current) {
      return;
    }

    // Debounce saves to prevent excessive writes
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(() => {
      saveToLocalStorage();
      if (isAuthenticated && user) {
        saveToFirestore();
      }
    }, 500);

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [progress, bookmarks, selectedTopicType, isLoaded, isAuthenticated, user]);

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

  const saveToLocalStorage = async () => {
    const state: AppState = {
      progress,
      bookmarks,
      currentDeckIndex: 0,
      selectedTopicType,
    };

    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
      console.error('Failed to save local state:', error);
    }
  };

  const saveToFirestore = async () => {
    if (!user) return;

    setHasPendingChanges(true);
    try {
      const userDocRef = doc(db, 'users', user.uid);
      await setDoc(userDocRef, {
        progress,
        bookmarks,
        selectedTopicType,
        updatedAt: new Date().toISOString(),
      }, { merge: true });
      setSyncError(null);
      setHasPendingChanges(false);
    } catch (error: any) {
      console.error('Failed to save to Firestore:', error);
      if (error?.code === 'firestore/unavailable') {
        setSyncError('Offline - changes will sync when connected');
      } else {
        setSyncError('Sync pending');
      }
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
