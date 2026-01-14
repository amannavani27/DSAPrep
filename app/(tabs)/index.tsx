import React, { useRef, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Dimensions, ScrollView } from 'react-native';
import { DeckSwiper, DeckSwiperRef } from '../../components/DeckSwiper';
import { SwipeCard } from '../../components/SwipeCard';
import { useProgress } from '../../context/ProgressContext';
import { Topic, TopicType } from '../../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function StudyScreen() {
  const {
    setSelectedTopicType,
    selectedTopicType,
    dsaStats,
    systemDesignStats,
    markAsKnown,
    markForReview,
    getTopicStatus,
    stats,
    getTopicsForType,
  } = useProgress();

  const [isStudying, setIsStudying] = useState(false);
  const [cardIndex, setCardIndex] = useState(0);
  const [deckFinished, setDeckFinished] = useState(false);
  const [deck, setDeck] = useState<Topic[]>([]);
  const swiperRef = useRef<DeckSwiperRef>(null);

  const topics = getTopicsForType(selectedTopicType);

  // Build deck only when starting to study (not on every progress change)
  const buildDeck = (type: TopicType) => {
    const topicsForType = getTopicsForType(type);
    const reviewTopics = topicsForType.filter(
      (t) => getTopicStatus(t.id) === 'review'
    );
    const unseenTopics = topicsForType.filter(
      (t) => getTopicStatus(t.id) === 'unseen'
    );
    return [...reviewTopics, ...unseenTopics];
  };

  const handleSelectSection = (type: TopicType) => {
    setSelectedTopicType(type);
    setCardIndex(0);
    setDeckFinished(false);
    setDeck(buildDeck(type));
    setIsStudying(true);
  };

  const handleBackToHome = () => {
    setIsStudying(false);
    setDeckFinished(false);
    setCardIndex(0);
  };

  const handleSwipedLeft = (index: number) => {
    const topic = deck[index];
    if (topic) {
      markAsKnown(topic.id);
    }
  };

  const handleSwipedRight = (index: number) => {
    const topic = deck[index];
    if (topic) {
      markForReview(topic.id);
    }
  };

  const handleSwipedAll = () => {
    setDeckFinished(true);
  };

  const restartDeck = () => {
    setCardIndex(0);
    setDeckFinished(false);
    setDeck(buildDeck(selectedTopicType));
  };

  // Home screen with section selection
  if (!isStudying) {
    return (
      <ScrollView style={styles.container} contentContainerStyle={styles.selectorContent}>
        <View style={styles.headerContainer}>
          <Text style={styles.headerTitle}>Interview Prep</Text>
          <Text style={styles.headerSubtitle}>Choose a topic to study</Text>
        </View>

        <TouchableOpacity
          style={styles.topicCard}
          onPress={() => handleSelectSection('dsa')}
          activeOpacity={0.8}
        >
          <View style={styles.topicCardHeader}>
            <View style={styles.topicIconContainer}>
              <Text style={styles.topicIcon}>{'</>'}</Text>
            </View>
            <View style={styles.topicCardBadge}>
              <Text style={styles.topicCardBadgeText}>{dsaStats.total} Topics</Text>
            </View>
          </View>
          <Text style={styles.topicCardTitle}>Data Structures & Algorithms</Text>
          <Text style={styles.topicCardDescription}>
            Arrays, Trees, Graphs, Dynamic Programming, and more. Master the fundamentals of coding interviews.
          </Text>
          <View style={styles.topicCardProgress}>
            <View style={styles.progressTrack}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${dsaStats.percentComplete}%` },
                ]}
              />
            </View>
            <View style={styles.topicCardStats}>
              <Text style={styles.progressPercent}>{dsaStats.percentComplete}% Complete</Text>
              <Text style={styles.statsDetail}>
                {dsaStats.known} mastered | {dsaStats.review} review
              </Text>
            </View>
          </View>
          <View style={styles.startButton}>
            <Text style={styles.startButtonText}>Start Studying</Text>
            <Text style={styles.startButtonArrow}>{'\u2192'}</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.topicCard, styles.topicCardAlt]}
          onPress={() => handleSelectSection('systemDesign')}
          activeOpacity={0.8}
        >
          <View style={styles.topicCardHeader}>
            <View style={[styles.topicIconContainer, styles.topicIconContainerAlt]}>
              <Text style={[styles.topicIcon, styles.topicIconAlt]}>{'{ }'}</Text>
            </View>
            <View style={[styles.topicCardBadge, styles.topicCardBadgeAlt]}>
              <Text style={[styles.topicCardBadgeText, styles.topicCardBadgeTextAlt]}>{systemDesignStats.total} Topics</Text>
            </View>
          </View>
          <Text style={styles.topicCardTitle}>System Design</Text>
          <Text style={styles.topicCardDescription}>
            Scalability, databases, caching, microservices, and real-world system architectures for senior interviews.
          </Text>
          <View style={styles.topicCardProgress}>
            <View style={styles.progressTrack}>
              <View
                style={[
                  styles.progressFillAlt,
                  { width: `${systemDesignStats.percentComplete}%` },
                ]}
              />
            </View>
            <View style={styles.topicCardStats}>
              <Text style={styles.progressPercentAlt}>{systemDesignStats.percentComplete}% Complete</Text>
              <Text style={styles.statsDetail}>
                {systemDesignStats.known} mastered | {systemDesignStats.review} review
              </Text>
            </View>
          </View>
          <View style={[styles.startButton, styles.startButtonAlt]}>
            <Text style={styles.startButtonText}>Start Studying</Text>
            <Text style={styles.startButtonArrow}>{'\u2192'}</Text>
          </View>
        </TouchableOpacity>
      </ScrollView>
    );
  }

  const topicTypeLabel = selectedTopicType === 'dsa'
    ? 'Data Structures & Algorithms'
    : 'System Design';

  const accentColor = selectedTopicType === 'dsa' ? '#e94560' : '#3b82f6';

  // Completed all cards screen
  if (deck.length === 0 || deckFinished) {
    return (
      <View style={styles.container}>
        <View style={styles.emptyState}>
          <Text style={styles.emptyEmoji}>{'\uD83C\uDF89'}</Text>
          <Text style={styles.emptyTitle}>All Done!</Text>
          <Text style={styles.emptySubtitle}>
            You've reviewed all {topics.length} {topicTypeLabel} topics.
          </Text>
          <Text style={[styles.statsText, { color: accentColor }]}>
            {stats.known} mastered | {stats.review} for review
          </Text>
          <TouchableOpacity style={[styles.restartButton, { backgroundColor: accentColor }]} onPress={restartDeck}>
            <Text style={styles.restartButtonText}>Review Again</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.backButton} onPress={handleBackToHome}>
            <Text style={styles.backButtonText}>{'\u2190'} Back to Topics</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // Study deck screen
  return (
    <View style={styles.container}>
      <View style={styles.studyHeader}>
        <TouchableOpacity onPress={handleBackToHome} style={styles.backButtonSmall}>
          <Text style={[styles.backButtonSmallText, { color: accentColor }]}>{'\u2190'} Back</Text>
        </TouchableOpacity>
        <Text style={styles.studyHeaderTitle}>{topicTypeLabel}</Text>
      </View>

      <View style={styles.progressBar}>
        <View style={styles.progressText}>
          <Text style={styles.progressLabel}>Progress</Text>
          <Text style={[styles.progressValue, { color: accentColor }]}>{stats.percentComplete}%</Text>
        </View>
        <View style={styles.progressTrack}>
          <View
            style={[styles.progressFill, { width: `${stats.percentComplete}%`, backgroundColor: accentColor }]}
          />
        </View>
      </View>

      <View style={styles.swiperContainer}>
        <DeckSwiper
          key={`${selectedTopicType}-${deck.length}`}
          ref={swiperRef}
          cards={deck}
          cardIndex={cardIndex}
          renderCard={(topic) =>
            topic ? <SwipeCard topic={topic} /> : null
          }
          onSwipedLeft={handleSwipedLeft}
          onSwipedRight={handleSwipedRight}
          onSwipedAll={handleSwipedAll}
          stackSize={3}
          overlayLabels={{
            left: {
              title: 'KNOW IT',
              style: {
                label: styles.overlayLabelLeft,
                wrapper: styles.overlayWrapperLeft,
              },
            },
            right: {
              title: 'REVIEW',
              style: {
                label: [styles.overlayLabelRight, { backgroundColor: accentColor }],
                wrapper: styles.overlayWrapperRight,
              },
            },
          }}
        />
      </View>

      <View style={styles.buttonRow}>
        <TouchableOpacity
          style={[styles.actionButton, styles.knowButton]}
          onPress={() => swiperRef.current?.swipeLeft()}
        >
          <Text style={styles.buttonIcon}>{'\u2713'}</Text>
          <Text style={styles.buttonLabel}>Know It</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionButton, { backgroundColor: accentColor }]}
          onPress={() => swiperRef.current?.swipeRight()}
        >
          <Text style={styles.buttonIcon}>{'\u21BB'}</Text>
          <Text style={styles.buttonLabel}>Review</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
  },
  selectorContent: {
    padding: 20,
    paddingBottom: 40,
  },
  headerContainer: {
    marginBottom: 24,
    marginTop: 10,
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#6c7086',
  },
  topicCard: {
    backgroundColor: '#16213e',
    borderRadius: 20,
    padding: 24,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#e94560',
  },
  topicCardAlt: {
    borderColor: '#3b82f6',
  },
  topicCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  topicIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 12,
    backgroundColor: 'rgba(233, 69, 96, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  topicIconContainerAlt: {
    backgroundColor: 'rgba(59, 130, 246, 0.15)',
  },
  topicIcon: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#e94560',
  },
  topicIconAlt: {
    color: '#3b82f6',
  },
  topicCardBadge: {
    backgroundColor: 'rgba(233, 69, 96, 0.15)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  topicCardBadgeAlt: {
    backgroundColor: 'rgba(59, 130, 246, 0.15)',
  },
  topicCardBadgeText: {
    color: '#e94560',
    fontSize: 12,
    fontWeight: '600',
  },
  topicCardBadgeTextAlt: {
    color: '#3b82f6',
  },
  topicCardTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  topicCardDescription: {
    fontSize: 14,
    color: '#a6adc8',
    lineHeight: 20,
    marginBottom: 20,
  },
  topicCardProgress: {
    marginBottom: 20,
  },
  topicCardStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
  },
  progressPercent: {
    fontSize: 14,
    fontWeight: '600',
    color: '#e94560',
  },
  progressPercentAlt: {
    fontSize: 14,
    fontWeight: '600',
    color: '#3b82f6',
  },
  statsDetail: {
    fontSize: 12,
    color: '#6c7086',
  },
  startButton: {
    backgroundColor: '#e94560',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 14,
    borderRadius: 12,
    gap: 8,
  },
  startButtonAlt: {
    backgroundColor: '#3b82f6',
  },
  startButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  startButtonArrow: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  studyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 8,
  },
  backButtonSmall: {
    paddingVertical: 6,
    paddingRight: 12,
  },
  backButtonSmallText: {
    fontSize: 14,
    fontWeight: '500',
  },
  studyHeaderTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    flex: 1,
  },
  progressBar: {
    paddingHorizontal: 20,
    paddingTop: 4,
  },
  progressText: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  progressLabel: {
    color: '#6c7086',
    fontSize: 12,
  },
  progressValue: {
    fontSize: 12,
    fontWeight: '600',
  },
  progressTrack: {
    height: 4,
    backgroundColor: '#1a1a2e',
    borderRadius: 2,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#e94560',
    borderRadius: 2,
  },
  progressFillAlt: {
    height: '100%',
    backgroundColor: '#3b82f6',
    borderRadius: 2,
  },
  swiperContainer: {
    flex: 1,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 20,
    paddingBottom: 20,
    paddingHorizontal: 40,
  },
  actionButton: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 14,
    borderRadius: 16,
  },
  knowButton: {
    backgroundColor: '#1db954',
  },
  buttonIcon: {
    fontSize: 20,
    color: '#fff',
    marginBottom: 2,
  },
  buttonLabel: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  overlayLabelLeft: {
    backgroundColor: '#1db954',
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
    padding: 10,
    borderRadius: 8,
  },
  overlayLabelRight: {
    backgroundColor: '#e94560',
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
    padding: 10,
    borderRadius: 8,
  },
  overlayWrapperLeft: {
    flexDirection: 'column',
    alignItems: 'flex-end',
    justifyContent: 'flex-start',
    marginTop: 20,
    marginLeft: -20,
  },
  overlayWrapperRight: {
    flexDirection: 'column',
    alignItems: 'flex-start',
    justifyContent: 'flex-start',
    marginTop: 20,
    marginLeft: 20,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyEmoji: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 16,
    color: '#6c7086',
    textAlign: 'center',
    marginBottom: 12,
  },
  statsText: {
    fontSize: 14,
    marginBottom: 24,
  },
  restartButton: {
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 12,
    marginBottom: 12,
  },
  restartButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  backButton: {
    paddingHorizontal: 32,
    paddingVertical: 14,
  },
  backButtonText: {
    color: '#6c7086',
    fontSize: 14,
  },
});
