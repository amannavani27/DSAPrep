import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';

interface CodeBlockProps {
  code: string;
  language: string;
}

export function CodeBlock({ code, language }: CodeBlockProps) {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.language}>{language}</Text>
      </View>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.scrollView}
      >
        <ScrollView
          nestedScrollEnabled
          showsVerticalScrollIndicator={false}
          style={styles.codeScrollView}
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
  scrollView: {
    maxHeight: 200,
  },
  codeScrollView: {
    padding: 12,
  },
  code: {
    fontFamily: 'Menlo',
    fontSize: 12,
    color: '#cdd6f4',
    lineHeight: 18,
  },
});
