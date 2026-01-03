import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Modal,
  Dimensions,
} from 'react-native';
import { useProgress } from '../../context/ProgressContext';
import { TopicType, Topic } from '../../types';
import { CodeBlock } from '../../components/CodeBlock';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

export default function ProgressScreen() {
  const {
    dsaStats,
    systemDesignStats,
    getTopicStatus,
    resetProgress,
    getTopicsForType,
    isBookmarked,
    toggleBookmark,
  } = useProgress();

  const [activeTab, setActiveTab] = useState<TopicType>('dsa');
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [modalVisible, setModalVisible] = useState(false);

  const currentStats = activeTab === 'dsa' ? dsaStats : systemDesignStats;
  const currentTopics = getTopicsForType(activeTab);

  const knownTopics = currentTopics.filter((t) => getTopicStatus(t.id) === 'known');
  const reviewTopics = currentTopics.filter((t) => getTopicStatus(t.id) === 'review');
  const unseenTopics = currentTopics.filter((t) => getTopicStatus(t.id) === 'unseen');

  const handleTopicPress = (topic: Topic) => {
    setSelectedTopic(topic);
    setModalVisible(true);
  };

  const handleReset = () => {
    const topicTypeName = activeTab === 'dsa' ? 'DSA' : 'System Design';
    Alert.alert(
      'Reset Progress',
      `Are you sure you want to reset all ${topicTypeName} progress?\n\nThis will clear all mastered topics, review items, and bookmarks for ${topicTypeName}.\n\nThis action cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Yes, Reset Everything',
          style: 'destructive',
          onPress: () => {
            Alert.alert(
              'Final Confirmation',
              `This will permanently delete your ${topicTypeName} progress. Are you absolutely sure?`,
              [
                { text: 'No, Keep My Progress', style: 'cancel' },
                {
                  text: 'Yes, Delete It',
                  style: 'destructive',
                  onPress: () => resetProgress(activeTab),
                },
              ]
            );
          },
        },
      ]
    );
  };

  const tabColor = activeTab === 'dsa' ? '#e94560' : '#3b82f6';

  const TopicItem = ({ topic, status }: { topic: Topic; status: 'known' | 'review' | 'unseen' }) => (
    <TouchableOpacity
      style={styles.topicItem}
      onPress={() => handleTopicPress(topic)}
      activeOpacity={0.7}
    >
      <View
        style={[
          styles.statusDot,
          {
            backgroundColor:
              status === 'known' ? '#1db954' : status === 'review' ? tabColor : '#6c7086',
          },
        ]}
      />
      <View style={styles.topicInfo}>
        <Text style={styles.topicTitle}>{topic.title}</Text>
        <Text style={styles.topicCategory}>{topic.category}</Text>
      </View>
      {status === 'known' && <Text style={styles.checkmark}>{'\u2713'}</Text>}
      {status === 'review' && <Text style={[styles.reviewIcon, { color: tabColor }]}>{'\u21BB'}</Text>}
      <Text style={styles.chevron}>{'\u203A'}</Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.tabContainer}>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'dsa' && styles.activeTab]}
            onPress={() => setActiveTab('dsa')}
          >
            <Text style={[styles.tabText, activeTab === 'dsa' && styles.activeTabText]}>
              DSA
            </Text>
            <Text style={[styles.tabSubtext, activeTab === 'dsa' && styles.activeTabSubtext]}>
              {dsaStats.percentComplete}%
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'systemDesign' && styles.activeTabAlt]}
            onPress={() => setActiveTab('systemDesign')}
          >
            <Text style={[styles.tabText, activeTab === 'systemDesign' && styles.activeTabTextAlt]}>
              System Design
            </Text>
            <Text style={[styles.tabSubtext, activeTab === 'systemDesign' && styles.activeTabSubtextAlt]}>
              {systemDesignStats.percentComplete}%
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.statsCard}>
          <View style={[styles.progressCircle, { borderColor: tabColor }]}>
            <Text style={styles.progressPercent}>{currentStats.percentComplete}%</Text>
            <Text style={styles.progressLabel}>Complete</Text>
          </View>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Text style={[styles.statNumber, { color: '#1db954' }]}>
                {currentStats.known}
              </Text>
              <Text style={styles.statLabel}>Mastered</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={[styles.statNumber, { color: tabColor }]}>
                {currentStats.review}
              </Text>
              <Text style={styles.statLabel}>Review</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={[styles.statNumber, { color: '#6c7086' }]}>
                {currentStats.unseen}
              </Text>
              <Text style={styles.statLabel}>Unseen</Text>
            </View>
          </View>
        </View>

        {knownTopics.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Mastered Topics ({knownTopics.length})</Text>
            {knownTopics.map((topic) => (
              <TopicItem key={topic.id} topic={topic} status="known" />
            ))}
          </View>
        )}

        {reviewTopics.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Needs Review ({reviewTopics.length})</Text>
            {reviewTopics.map((topic) => (
              <TopicItem key={topic.id} topic={topic} status="review" />
            ))}
          </View>
        )}

        {unseenTopics.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Not Yet Seen ({unseenTopics.length})</Text>
            {unseenTopics.map((topic) => (
              <TopicItem key={topic.id} topic={topic} status="unseen" />
            ))}
          </View>
        )}

        <View style={styles.resetSection}>
          <Text style={styles.resetWarning}>
            Reset will clear all progress for {activeTab === 'dsa' ? 'DSA' : 'System Design'}
          </Text>
          <TouchableOpacity
            style={[styles.resetButton, { borderColor: tabColor }]}
            onPress={handleReset}
          >
            <Text style={[styles.resetButtonText, { color: tabColor }]}>
              Reset {activeTab === 'dsa' ? 'DSA' : 'System Design'} Progress
            </Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      {/* Topic Detail Modal */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <TouchableOpacity
                style={styles.closeButton}
                onPress={() => setModalVisible(false)}
              >
                <Text style={styles.closeButtonText}>{'\u2715'}</Text>
              </TouchableOpacity>
            </View>

            {selectedTopic && (
              <ScrollView style={styles.modalScroll} showsVerticalScrollIndicator={false}>
                <View style={styles.modalCardHeader}>
                  <View
                    style={[
                      styles.modalCategoryBadge,
                      { backgroundColor: selectedTopic.topicType === 'dsa' ? '#e94560' : '#3b82f6' },
                    ]}
                  >
                    <Text style={styles.modalCategoryText}>{selectedTopic.category}</Text>
                  </View>
                  <TouchableOpacity
                    onPress={() => toggleBookmark(selectedTopic.id)}
                    style={styles.bookmarkButton}
                  >
                    <Text style={styles.bookmarkIcon}>
                      {isBookmarked(selectedTopic.id) ? '\u2605' : '\u2606'}
                    </Text>
                  </TouchableOpacity>
                </View>

                <Text style={styles.modalTitle}>{selectedTopic.title}</Text>

                <View
                  style={[
                    styles.statusBadge,
                    {
                      backgroundColor:
                        getTopicStatus(selectedTopic.id) === 'known'
                          ? '#1db954'
                          : getTopicStatus(selectedTopic.id) === 'review'
                          ? tabColor
                          : '#6c7086',
                    },
                  ]}
                >
                  <Text style={styles.statusText}>
                    {getTopicStatus(selectedTopic.id) === 'known'
                      ? 'Mastered'
                      : getTopicStatus(selectedTopic.id) === 'review'
                      ? 'Needs Review'
                      : 'Not Yet Seen'}
                  </Text>
                </View>

                <Text style={styles.modalDescription}>{selectedTopic.description}</Text>

                <View style={styles.keyPointsContainer}>
                  <Text
                    style={[
                      styles.modalSectionTitle,
                      { color: selectedTopic.topicType === 'dsa' ? '#e94560' : '#3b82f6' },
                    ]}
                  >
                    Key Points
                  </Text>
                  {selectedTopic.keyPoints.map((point, index) => (
                    <View key={index} style={styles.keyPoint}>
                      <Text
                        style={[
                          styles.bullet,
                          { color: selectedTopic.topicType === 'dsa' ? '#e94560' : '#3b82f6' },
                        ]}
                      >
                        {'\u2022'}
                      </Text>
                      <Text style={styles.keyPointText}>{point}</Text>
                    </View>
                  ))}
                </View>

                {selectedTopic.codeExample && (
                  <View style={styles.codeContainer}>
                    <Text
                      style={[
                        styles.modalSectionTitle,
                        { color: selectedTopic.topicType === 'dsa' ? '#e94560' : '#3b82f6' },
                      ]}
                    >
                      {selectedTopic.topicType === 'systemDesign' ? 'Example' : 'Code Example'}
                    </Text>
                    <CodeBlock
                      code={selectedTopic.codeExample}
                      language={selectedTopic.codeLanguage || 'plaintext'}
                    />
                  </View>
                )}
              </ScrollView>
            )}
          </View>
        </View>
      </Modal>
    </View>
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
    marginBottom: 20,
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    borderRadius: 10,
  },
  activeTab: {
    backgroundColor: 'rgba(233, 69, 96, 0.15)',
  },
  activeTabAlt: {
    backgroundColor: 'rgba(59, 130, 246, 0.15)',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6c7086',
  },
  activeTabText: {
    color: '#e94560',
  },
  activeTabTextAlt: {
    color: '#3b82f6',
  },
  tabSubtext: {
    fontSize: 11,
    color: '#6c7086',
    marginTop: 2,
  },
  activeTabSubtext: {
    color: '#e94560',
  },
  activeTabSubtextAlt: {
    color: '#3b82f6',
  },
  statsCard: {
    backgroundColor: '#16213e',
    borderRadius: 20,
    padding: 24,
    alignItems: 'center',
    marginBottom: 24,
  },
  progressCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#1a1a2e',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    borderWidth: 4,
  },
  progressPercent: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
  },
  progressLabel: {
    fontSize: 12,
    color: '#6c7086',
    marginTop: 2,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 28,
    fontWeight: 'bold',
  },
  statLabel: {
    fontSize: 12,
    color: '#6c7086',
    marginTop: 4,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  topicItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#16213e',
    borderRadius: 12,
    padding: 14,
    marginBottom: 8,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 12,
  },
  topicInfo: {
    flex: 1,
  },
  topicTitle: {
    fontSize: 15,
    color: '#fff',
    fontWeight: '500',
  },
  topicCategory: {
    fontSize: 12,
    color: '#6c7086',
    marginTop: 2,
  },
  checkmark: {
    fontSize: 18,
    color: '#1db954',
    marginRight: 8,
  },
  reviewIcon: {
    fontSize: 18,
    marginRight: 8,
  },
  chevron: {
    fontSize: 20,
    color: '#6c7086',
  },
  resetSection: {
    marginTop: 20,
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: '#1a1a2e',
  },
  resetWarning: {
    fontSize: 12,
    color: '#6c7086',
    textAlign: 'center',
    marginBottom: 12,
  },
  resetButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  resetButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#16213e',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    maxHeight: SCREEN_HEIGHT * 0.85,
    paddingBottom: 40,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    padding: 16,
    paddingBottom: 0,
  },
  closeButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#1a1a2e',
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  modalScroll: {
    paddingHorizontal: 20,
  },
  modalCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  modalCategoryBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  modalCategoryText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  bookmarkButton: {
    padding: 4,
  },
  bookmarkIcon: {
    fontSize: 24,
    color: '#ffd700',
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 12,
  },
  statusBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginBottom: 16,
  },
  statusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  modalDescription: {
    fontSize: 15,
    color: '#a6adc8',
    lineHeight: 22,
    marginBottom: 20,
  },
  keyPointsContainer: {
    marginBottom: 20,
  },
  modalSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  keyPoint: {
    flexDirection: 'row',
    marginBottom: 8,
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
    marginBottom: 20,
  },
});
