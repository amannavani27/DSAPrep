import React, { useRef, useImperativeHandle, forwardRef, useState, useCallback } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  Animated,
  PanResponder,
  GestureResponderEvent,
  PanResponderGestureState,
} from 'react-native';
import { useGesture } from '../context/GestureContext';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.25;

interface DeckSwiperProps<T> {
  cards: T[];
  cardIndex: number;
  renderCard: (card: T, index: number) => React.ReactNode;
  onSwipedLeft?: (index: number) => void;
  onSwipedRight?: (index: number) => void;
  onSwipedAll?: () => void;
  stackSize?: number;
  overlayLabels?: {
    left?: { title: string; style: { label: object; wrapper: object } };
    right?: { title: string; style: { label: object; wrapper: object } };
  };
}

export interface DeckSwiperRef {
  swipeLeft: () => void;
  swipeRight: () => void;
}

function DeckSwiperInner<T>(
  props: DeckSwiperProps<T>,
  ref: React.Ref<DeckSwiperRef>
) {
  const {
    cards,
    cardIndex: initialIndex,
    renderCard,
    onSwipedLeft,
    onSwipedRight,
    onSwipedAll,
    stackSize = 3,
    overlayLabels,
  } = props;

  const { shouldBlockSwipe } = useGesture();
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [activeCardIndex, setActiveCardIndex] = useState<number | null>(null);
  const currentIndexRef = useRef(currentIndex);
  const position = useRef(new Animated.ValueXY()).current;
  const isAnimating = useRef(false);

  // Keep ref in sync with state
  React.useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);

  const handleSwipeComplete = useCallback(
    (direction: 'left' | 'right') => {
      const idx = currentIndexRef.current;

      if (direction === 'left') {
        onSwipedLeft?.(idx);
      } else {
        onSwipedRight?.(idx);
      }

      const nextIndex = idx + 1;

      // Clear active card and update index first
      setActiveCardIndex(null);
      setCurrentIndex(nextIndex);
      currentIndexRef.current = nextIndex;

      // Reset position AFTER React processes state updates
      // This prevents the old card from snapping back to center before unmounting
      requestAnimationFrame(() => {
        position.setValue({ x: 0, y: 0 });
        isAnimating.current = false;
      });

      if (nextIndex >= cards.length) {
        onSwipedAll?.();
      }
    },
    [cards.length, onSwipedLeft, onSwipedRight, onSwipedAll, position]
  );

  const animateSwipe = useCallback(
    (direction: 'left' | 'right') => {
      if (isAnimating.current) return;
      isAnimating.current = true;

      const destination = direction === 'left' ? -SCREEN_WIDTH * 1.5 : SCREEN_WIDTH * 1.5;
      Animated.timing(position, {
        toValue: { x: destination, y: 0 },
        duration: 250,
        useNativeDriver: false,
      }).start(() => {
        handleSwipeComplete(direction);
      });
    },
    [position, handleSwipeComplete]
  );

  useImperativeHandle(ref, () => ({
    swipeLeft: () => animateSwipe('left'),
    swipeRight: () => animateSwipe('right'),
  }));

  const panResponder = React.useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => false,
        onMoveShouldSetPanResponder: (
          _: GestureResponderEvent,
          gestureState: PanResponderGestureState
        ) => {
          // Don't claim gesture if touch is inside CodeBlock
          if (shouldBlockSwipe.current) {
            return false;
          }
          // Only respond to horizontal gestures, let vertical pass through for scrolling
          const { dx, dy } = gestureState;
          return Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 10;
        },
        onPanResponderGrant: () => {
          // Mark the current top card as active when drag starts
          setActiveCardIndex(currentIndexRef.current);
        },
        onPanResponderMove: (_, gestureState) => {
          if (!shouldBlockSwipe.current) {
            position.setValue({ x: gestureState.dx, y: 0 });
          }
        },
        onPanResponderRelease: (_, gestureState) => {
          if (shouldBlockSwipe.current) {
            setActiveCardIndex(null);
            return;
          }
          if (gestureState.dx < -SWIPE_THRESHOLD) {
            animateSwipe('left');
          } else if (gestureState.dx > SWIPE_THRESHOLD) {
            animateSwipe('right');
          } else {
            Animated.spring(position, {
              toValue: { x: 0, y: 0 },
              useNativeDriver: false,
              friction: 5,
            }).start(() => {
              setActiveCardIndex(null);
            });
          }
        },
      }),
    [shouldBlockSwipe, position, animateSwipe]
  );

  const getCardStyle = () => {
    const rotate = position.x.interpolate({
      inputRange: [-SCREEN_WIDTH, 0, SCREEN_WIDTH],
      outputRange: ['-15deg', '0deg', '15deg'],
      extrapolate: 'clamp',
    });

    return {
      transform: [
        { translateX: position.x },
        { rotate },
      ],
    };
  };

  const getLeftOverlayStyle = () => {
    const opacity = position.x.interpolate({
      inputRange: [-SWIPE_THRESHOLD, 0],
      outputRange: [1, 0],
      extrapolate: 'clamp',
    });
    return { opacity };
  };

  const getRightOverlayStyle = () => {
    const opacity = position.x.interpolate({
      inputRange: [0, SWIPE_THRESHOLD],
      outputRange: [0, 1],
      extrapolate: 'clamp',
    });
    return { opacity };
  };

  // Animated scale for the next card (scales up as top card is dragged away)
  const getNextCardStyle = () => {
    const scale = position.x.interpolate({
      inputRange: [-SCREEN_WIDTH / 2, 0, SCREEN_WIDTH / 2],
      outputRange: [1, 0.95, 1],
      extrapolate: 'clamp',
    });
    const translateY = position.x.interpolate({
      inputRange: [-SCREEN_WIDTH / 2, 0, SCREEN_WIDTH / 2],
      outputRange: [0, 12, 0],
      extrapolate: 'clamp',
    });
    return {
      transform: [{ scale }, { translateY }],
    };
  };

  const renderCards = () => {
    if (currentIndex >= cards.length) {
      return null;
    }

    const visibleCards = [];

    // Render the stack of cards (back to front)
    for (let i = Math.min(stackSize - 1, cards.length - currentIndex - 1); i >= 0; i--) {
      const cardIdx = currentIndex + i;
      const card = cards[cardIdx];
      const isTopCard = i === 0;
      const isSecondCard = i === 1;
      const isActive = activeCardIndex === cardIdx;

      if (isTopCard) {
        // Top card - only animate if it's the active card being dragged
        visibleCards.push(
          <Animated.View
            key={cardIdx}
            style={[
              styles.cardContainer,
              isActive ? getCardStyle() : {},
              { zIndex: stackSize }
            ]}
            {...panResponder.panHandlers}
          >
            {renderCard(card, cardIdx)}
            {isActive && overlayLabels?.left && (
              <Animated.View style={[styles.overlayContainer, overlayLabels.left.style.wrapper, getLeftOverlayStyle()]}>
                <Animated.Text style={overlayLabels.left.style.label}>
                  {overlayLabels.left.title}
                </Animated.Text>
              </Animated.View>
            )}
            {isActive && overlayLabels?.right && (
              <Animated.View style={[styles.overlayContainer, overlayLabels.right.style.wrapper, getRightOverlayStyle()]}>
                <Animated.Text style={overlayLabels.right.style.label}>
                  {overlayLabels.right.title}
                </Animated.Text>
              </Animated.View>
            )}
          </Animated.View>
        );
      } else if (isSecondCard) {
        // Second card - scales up as top card is dragged (only when there's an active drag)
        visibleCards.push(
          <Animated.View
            key={cardIdx}
            style={[
              styles.cardContainer,
              activeCardIndex !== null ? getNextCardStyle() : { transform: [{ scale: 0.95 }, { translateY: 12 }] },
              { zIndex: stackSize - 1 }
            ]}
          >
            {renderCard(card, cardIdx)}
          </Animated.View>
        );
      } else {
        // Other cards in stack - static
        const stackStyle = {
          transform: [
            { scale: 1 - i * 0.05 },
            { translateY: i * 12 },
          ],
          zIndex: stackSize - i,
        };
        visibleCards.push(
          <View key={cardIdx} style={[styles.cardContainer, stackStyle]}>
            {renderCard(card, cardIdx)}
          </View>
        );
      }
    }

    return visibleCards;
  };

  return <View style={styles.container}>{renderCards()}</View>;
}

export const DeckSwiper = forwardRef(DeckSwiperInner) as <T>(
  props: DeckSwiperProps<T> & { ref?: React.Ref<DeckSwiperRef> }
) => React.ReactElement;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardContainer: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  overlayContainer: {
    position: 'absolute',
    top: 40,
  },
});
