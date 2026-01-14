import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  ScrollView,
} from 'react-native';
import { Topic } from '../types';
import { CodeBlock } from './CodeBlock';
import { useProgress } from '../context/ProgressContext';

interface SwipeCardProps {
  topic: Topic;
}

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CARD_WIDTH = SCREEN_WIDTH - 40;

export function SwipeCard({ topic }: SwipeCardProps) {
  const { isBookmarked, toggleBookmark } = useProgress();
  const bookmarked = isBookmarked(topic.id);

  const isSystemDesign = topic.topicType === 'systemDesign';
  const accentColor = isSystemDesign ? '#3b82f6' : '#e94560';

  return (
    <View style={styles.card}>
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={true}
        contentContainerStyle={styles.scrollContent}
        nestedScrollEnabled={true}
        bounces={true}
      >
        <View style={styles.header}>
          <View style={styles.badgeRow}>
            <View style={[styles.categoryBadge, { backgroundColor: accentColor }]}>
              <Text style={styles.categoryText}>{topic.category}</Text>
            </View>
            {isSystemDesign && (
              <View style={[styles.typeBadge, { backgroundColor: 'rgba(59, 130, 246, 0.15)' }]}>
                <Text style={[styles.typeText, { color: accentColor }]}>SD</Text>
              </View>
            )}
          </View>
          <TouchableOpacity
            onPress={() => toggleBookmark(topic.id)}
            style={styles.bookmarkButton}
          >
            <Text style={styles.bookmarkIcon}>{bookmarked ? '\u2605' : '\u2606'}</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.title}>{topic.title}</Text>
        <Text style={styles.description}>{topic.description}</Text>

        <View style={styles.keyPointsContainer}>
          <Text style={[styles.sectionTitle, { color: accentColor }]}>Key Points</Text>
          {topic.keyPoints.map((point, index) => (
            <View key={index} style={styles.keyPoint}>
              <Text style={[styles.bullet, { color: accentColor }]}>{'\u2022'}</Text>
              <Text style={styles.keyPointText}>{point}</Text>
            </View>
          ))}
        </View>

        {topic.codeExample && (
          <View style={styles.codeContainer}>
            <Text style={[styles.sectionTitle, { color: accentColor }]}>
              {isSystemDesign ? 'Example' : 'Code Example'}
            </Text>
            <CodeBlock code={topic.codeExample} language={topic.codeLanguage || 'plaintext'} />
          </View>
        )}
      </ScrollView>

      <View style={styles.swipeHint}>
        <Text style={styles.swipeHintText}>{'\u2190'} Know it</Text>
        <Text style={styles.swipeHintText}>Review {'\u2192'}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    width: CARD_WIDTH,
    height: '85%',
    backgroundColor: '#16213e',
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
    overflow: 'hidden',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  badgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  categoryBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  categoryText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  typeBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  typeText: {
    fontSize: 10,
    fontWeight: '600',
  },
  bookmarkButton: {
    padding: 4,
  },
  bookmarkIcon: {
    fontSize: 24,
    color: '#ffd700',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 12,
  },
  description: {
    fontSize: 15,
    color: '#a6adc8',
    lineHeight: 22,
    marginBottom: 16,
  },
  keyPointsContainer: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  keyPoint: {
    flexDirection: 'row',
    marginBottom: 6,
  },
  bullet: {
    marginRight: 8,
    fontSize: 14,
  },
  keyPointText: {
    color: '#cdd6f4',
    fontSize: 14,
    flex: 1,
    lineHeight: 20,
  },
  codeContainer: {
    marginBottom: 10,
  },
  swipeHint: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#2d2d44',
  },
  swipeHintText: {
    color: '#6c7086',
    fontSize: 12,
  },
});
