export type TopicType = 'dsa' | 'systemDesign';

export interface Topic {
  id: string;
  title: string;
  category: string;
  description: string;
  keyPoints: string[];
  codeExample?: string;
  codeLanguage?: 'python' | 'javascript' | 'java' | 'plaintext';
  topicType: TopicType;
}

// Alias for backwards compatibility
export type DSATopic = Topic;

export interface TopicProgress {
  topicId: string;
  status: 'known' | 'review' | 'unseen';
  lastSeen: number;
  reviewCount: number;
}

export interface AppState {
  progress: Record<string, TopicProgress>;
  bookmarks: string[];
  currentDeckIndex: number;
  selectedTopicType: TopicType;
}
