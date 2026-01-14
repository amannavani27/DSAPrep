import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { useProgress } from '../../context/ProgressContext';
import { CodeBlock } from '../../components/CodeBlock';
import { TopicType } from '../../types';

export default function BookmarksScreen() {
  const { bookmarks, toggleBookmark, getTopicStatus, allTopics } = useProgress();
  const [activeTab, setActiveTab] = useState<TopicType | 'all'>('all');

  // Sort by bookmark order (most recent first - newer bookmarks are at the end of the array)
  const bookmarkedTopics = allTopics
    .filter((t) => bookmarks.includes(t.id))
    .sort((a, b) => bookmarks.indexOf(b.id) - bookmarks.indexOf(a.id));

  const filteredBookmarks = activeTab === 'all'
    ? bookmarkedTopics
    : bookmarkedTopics.filter((t) => t.topicType === activeTab);

  const dsaBookmarkCount = bookmarkedTopics.filter((t) => t.topicType === 'dsa').length;
  const sdBookmarkCount = bookmarkedTopics.filter((t) => t.topicType === 'systemDesign').length;

  if (bookmarkedTopics.length === 0) {
    return (
      <View style={styles.container}>
        <View style={styles.emptyState}>
          <Text style={styles.emptyEmoji}>{'⭐'}</Text>
          <Text style={styles.emptyTitle}>No Bookmarks Yet</Text>
          <Text style={styles.emptySubtitle}>
            Tap the star on any card to save it here for quick reference.
          </Text>
        </View>
      </View>
    );
  }

  const getTabColor = () => {
    if (activeTab === 'dsa') return '#e94560';
    if (activeTab === 'systemDesign') return '#3b82f6';
    return '#a855f7';
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[
            styles.tab,
            activeTab === 'all' && styles.activeTabAll,
          ]}
          onPress={() => setActiveTab('all')}
        >
          <Text
            style={[
              styles.tabText,
              activeTab === 'all' && styles.activeTabTextAll,
            ]}
          >
            All
          </Text>
          <Text
            style={[
              styles.tabCount,
              activeTab === 'all' && styles.activeTabCountAll,
            ]}
          >
            {bookmarkedTopics.length}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.tab,
            activeTab === 'dsa' && styles.activeTab,
          ]}
          onPress={() => setActiveTab('dsa')}
        >
          <Text
            style={[
              styles.tabText,
              activeTab === 'dsa' && styles.activeTabText,
            ]}
          >
            DSA
          </Text>
          <Text
            style={[
              styles.tabCount,
              activeTab === 'dsa' && styles.activeTabCount,
            ]}
          >
            {dsaBookmarkCount}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.tab,
            activeTab === 'systemDesign' && styles.activeTabAlt,
          ]}
          onPress={() => setActiveTab('systemDesign')}
        >
          <Text
            style={[
              styles.tabText,
              activeTab === 'systemDesign' && styles.activeTabTextAlt,
            ]}
          >
            Sys Design
          </Text>
          <Text
            style={[
              styles.tabCount,
              activeTab === 'systemDesign' && styles.activeTabCountAlt,
            ]}
          >
            {sdBookmarkCount}
          </Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.header}>
        {filteredBookmarks.length} Bookmarked Topic
        {filteredBookmarks.length !== 1 ? 's' : ''}
      </Text>

      {filteredBookmarks.map((topic) => {
        const status = getTopicStatus(topic.id);
        const isSystemDesign = topic.topicType === 'systemDesign';
        const accentColor = isSystemDesign ? '#3b82f6' : '#e94560';

        return (
          <View key={topic.id} style={styles.card}>
            <View style={styles.cardHeader}>
              <View style={styles.badgeRow}>
                <View style={[styles.categoryBadge, { backgroundColor: accentColor }]}>
                  <Text style={styles.categoryText}>{topic.category}</Text>
                </View>
                <View style={[styles.typeBadge, { backgroundColor: isSystemDesign ? 'rgba(59, 130, 246, 0.15)' : 'rgba(233, 69, 96, 0.15)' }]}>
                  <Text style={[styles.typeText, { color: accentColor }]}>
                    {isSystemDesign ? 'System Design' : 'DSA'}
                  </Text>
                </View>
              </View>
              <View style={styles.statusContainer}>
                <View
                  style={[
                    styles.statusBadge,
                    {
                      backgroundColor:
                        status === 'known'
                          ? '#1db954'
                          : status === 'review'
                          ? accentColor
                          : '#6c7086',
                    },
                  ]}
                >
                  <Text style={styles.statusText}>
                    {status === 'known'
                      ? 'Mastered'
                      : status === 'review'
                      ? 'Review'
                      : 'Unseen'}
                  </Text>
                </View>
                <TouchableOpacity
                  onPress={() => toggleBookmark(topic.id)}
                  style={styles.bookmarkButton}
                >
                  <Text style={styles.bookmarkIcon}>{'★'}</Text>
                </TouchableOpacity>
              </View>
            </View>

            <Text style={styles.title}>{topic.title}</Text>
            <Text style={styles.description}>{topic.description}</Text>

            <View style={styles.keyPoints}>
              {topic.keyPoints.map((point, index) => (
                <View key={index} style={styles.keyPoint}>
                  <Text style={[styles.bullet, { color: accentColor }]}>{'•'}</Text>
                  <Text style={styles.keyPointText}>{point}</Text>
                </View>
              ))}
            </View>

            {topic.codeExample && (
              <CodeBlock code={topic.codeExample} language={topic.codeLanguage || 'plaintext'} />
            )}
          </View>
        );
      })}
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
    paddingBottom: 40,
  },
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: '#16213e',
    borderRadius: 12,
    padding: 4,
    marginBottom: 16,
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 10,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 6,
  },
  activeTab: {
    backgroundColor: 'rgba(233, 69, 96, 0.15)',
  },
  activeTabAlt: {
    backgroundColor: 'rgba(59, 130, 246, 0.15)',
  },
  activeTabAll: {
    backgroundColor: 'rgba(168, 85, 247, 0.15)',
  },
  tabText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6c7086',
  },
  activeTabText: {
    color: '#e94560',
  },
  activeTabTextAlt: {
    color: '#3b82f6',
  },
  activeTabTextAll: {
    color: '#a855f7',
  },
  tabCount: {
    fontSize: 11,
    color: '#6c7086',
    backgroundColor: '#1a1a2e',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 8,
    overflow: 'hidden',
  },
  activeTabCount: {
    color: '#e94560',
    backgroundColor: 'rgba(233, 69, 96, 0.2)',
  },
  activeTabCountAlt: {
    color: '#3b82f6',
    backgroundColor: 'rgba(59, 130, 246, 0.2)',
  },
  activeTabCountAll: {
    color: '#a855f7',
    backgroundColor: 'rgba(168, 85, 247, 0.2)',
  },
  header: {
    fontSize: 14,
    color: '#6c7086',
    marginBottom: 16,
  },
  card: {
    backgroundColor: '#16213e',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  badgeRow: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
    flex: 1,
  },
  categoryBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  categoryText: {
    color: '#fff',
    fontSize: 11,
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
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  statusText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },
  bookmarkButton: {
    padding: 4,
  },
  bookmarkIcon: {
    fontSize: 22,
    color: '#ffd700',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  description: {
    fontSize: 14,
    color: '#a6adc8',
    lineHeight: 20,
    marginBottom: 14,
  },
  keyPoints: {
    marginBottom: 14,
  },
  keyPoint: {
    flexDirection: 'row',
    marginBottom: 4,
  },
  bullet: {
    marginRight: 8,
    fontSize: 13,
  },
  keyPointText: {
    color: '#cdd6f4',
    fontSize: 13,
    flex: 1,
    lineHeight: 18,
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
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 15,
    color: '#6c7086',
    textAlign: 'center',
    lineHeight: 22,
  },
});
