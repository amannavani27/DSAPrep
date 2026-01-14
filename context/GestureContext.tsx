import React, { createContext, useContext, useRef } from 'react';

interface GestureContextType {
  shouldBlockSwipe: React.MutableRefObject<boolean>;
}

const GestureContext = createContext<GestureContextType | undefined>(undefined);

export function GestureProvider({ children }: { children: React.ReactNode }) {
  const shouldBlockSwipe = useRef(false);

  return (
    <GestureContext.Provider value={{ shouldBlockSwipe }}>
      {children}
    </GestureContext.Provider>
  );
}

export function useGesture() {
  const context = useContext(GestureContext);
  if (!context) {
    throw new Error('useGesture must be used within a GestureProvider');
  }
  return context;
}
