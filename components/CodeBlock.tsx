import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { useGesture } from '../context/GestureContext';

interface CodeBlockProps {
  code: string;
  language: string;
}

export function CodeBlock({ code, language }: CodeBlockProps) {
  const { shouldBlockSwipe } = useGesture();

  const handleTouchStart = () => {
    shouldBlockSwipe.current = true;
  };

  const handleTouchEnd = () => {
    // Small delay to ensure the gesture is complete
    setTimeout(() => {
      shouldBlockSwipe.current = false;
    }, 100);
  };

  return (
    <View
      style={styles.container}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      onTouchCancel={handleTouchEnd}
    >
      <View style={styles.header}>
        <Text style={styles.language}>{language}</Text>
      </View>
      <ScrollView
        style={styles.verticalScroll}
        nestedScrollEnabled={true}
        showsVerticalScrollIndicator={true}
        scrollEventThrottle={16}
      >
        <ScrollView
          horizontal={true}
          nestedScrollEnabled={true}
          showsHorizontalScrollIndicator={true}
          scrollEventThrottle={16}
          contentContainerStyle={styles.horizontalContent}
        >
          <Text style={styles.code}>{code}</Text>
        </ScrollView>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#1e1e2e',
    borderRadius: 12,
    overflow: 'hidden',
    marginTop: 12,
  },
  header: {
    backgroundColor: '#2d2d44',
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  language: {
    color: '#a6adc8',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  verticalScroll: {
    maxHeight: 200,
    padding: 12,
  },
  horizontalContent: {
    flexGrow: 1,
  },
  code: {
    fontFamily: 'Menlo',
    fontSize: 12,
    color: '#cdd6f4',
    lineHeight: 18,
  },
});
