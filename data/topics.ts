import { Topic } from '../types';

export const dsaTopics: Topic[] = [
  // ===== ARRAYS =====
  {
    id: 'two-pointer',
    title: 'Two Pointer Technique',
    category: 'Arrays',
    description: 'Use two pointers to traverse an array from different positions, often from both ends moving inward.',
    keyPoints: [
      'Great for sorted arrays',
      'O(n) time complexity',
      'Patterns: opposite ends, slow/fast',
      'Used in: Two Sum II, Container With Most Water',
    ],
    codeExample: `def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        curr = nums[left] + nums[right]
        if curr == target:
            return [left, right]
        elif curr < target:
            left += 1
        else:
            right -= 1
    return []`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'prefix-sum',
    title: 'Prefix Sum',
    category: 'Arrays',
    description: 'Precompute cumulative sums to answer range sum queries in O(1). Build prefix array where prefix[i] = sum of elements 0 to i-1.',
    keyPoints: [
      'O(n) preprocessing, O(1) queries',
      'Range sum = prefix[right+1] - prefix[left]',
      'Can extend to 2D matrices',
      'Used in: Subarray Sum Equals K, Range Sum Query',
    ],
    codeExample: `def range_sum(nums, queries):
    # Build prefix sum
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)

    # Answer queries in O(1)
    results = []
    for left, right in queries:
        results.append(prefix[right+1] - prefix[left])
    return results`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'kadane',
    title: "Kadane's Algorithm",
    category: 'Arrays',
    description: 'Find maximum sum contiguous subarray in O(n). Track current sum and reset when it goes negative.',
    keyPoints: [
      'O(n) time, O(1) space',
      'Reset current sum when negative',
      'Can track start/end indices',
      'Used in: Maximum Subarray, Max Circular Subarray',
    ],
    codeExample: `def max_subarray(nums):
    max_sum = curr_sum = nums[0]

    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)

    return max_sum`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'dutch-national-flag',
    title: 'Dutch National Flag',
    category: 'Arrays',
    description: 'Sort array with 3 distinct values in one pass using 3 pointers. Partition into low, mid, high regions.',
    keyPoints: [
      'O(n) time, O(1) space',
      'Three pointers: low, mid, high',
      'Single pass through array',
      'Used in: Sort Colors, Three-way Partition',
    ],
    codeExample: `def sort_colors(nums):
    low, mid, high = 0, 0, len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'merge-intervals',
    title: 'Merge Intervals',
    category: 'Arrays',
    description: 'Merge overlapping intervals by sorting by start time and combining when current overlaps with previous.',
    keyPoints: [
      'Sort intervals by start time first',
      'O(n log n) due to sorting',
      'Check if current.start <= prev.end',
      'Used in: Merge Intervals, Insert Interval',
    ],
    codeExample: `def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for curr in intervals[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1]:
            prev[1] = max(prev[1], curr[1])
        else:
            merged.append(curr)

    return merged`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'rotate-array',
    title: 'Rotate Array',
    category: 'Arrays',
    description: 'Rotate array by k positions using reversal algorithm. Reverse whole array, then reverse first k and last n-k elements.',
    keyPoints: [
      'O(n) time, O(1) space',
      'Three reversals: all, first k, last n-k',
      'Handle k > n with k %= n',
      'Used in: Rotate Array, Rotate String',
    ],
    codeExample: `def rotate(nums, k):
    n = len(nums)
    k %= n

    def reverse(l, r):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l, r = l + 1, r - 1

    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'next-permutation',
    title: 'Next Permutation',
    category: 'Arrays',
    description: 'Find next lexicographically greater permutation. Find rightmost ascending pair, swap with next larger, reverse suffix.',
    keyPoints: [
      'O(n) time, O(1) space',
      'Find pivot where nums[i] < nums[i+1]',
      'Swap with smallest larger element',
      'Used in: Next Permutation, Previous Permutation',
    ],
    codeExample: `def next_permutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    nums[i + 1:] = reversed(nums[i + 1:])`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'trapping-rain-water',
    title: 'Trapping Rain Water',
    category: 'Arrays',
    description: 'Calculate water trapped between bars. Water at each position = min(maxLeft, maxRight) - height.',
    keyPoints: [
      'Two pointer or DP approach',
      'O(n) time, O(1) space with two pointers',
      'Track leftMax and rightMax',
      'Used in: Trapping Rain Water, Container With Water',
    ],
    codeExample: `def trap(height):
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== STRINGS =====
  {
    id: 'sliding-window',
    title: 'Sliding Window',
    category: 'Strings',
    description: 'Maintain a window that slides through data. Expand right, shrink left based on conditions.',
    keyPoints: [
      'Fixed or variable window size',
      'O(n) time complexity',
      'Track state with hashmap/counter',
      'Used in: Longest Substring Without Repeating',
    ],
    codeExample: `def length_of_longest_substring(s):
    char_index = {}
    max_len = start = 0

    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        max_len = max(max_len, end - start + 1)

    return max_len`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'anagram-check',
    title: 'Anagram Detection',
    category: 'Strings',
    description: 'Check if two strings are anagrams using character frequency counting or sorting.',
    keyPoints: [
      'Counter comparison: O(n) time',
      'Sorting approach: O(n log n)',
      'Can use fixed-size array for lowercase',
      'Used in: Valid Anagram, Group Anagrams',
    ],
    codeExample: `from collections import Counter

def is_anagram(s, t):
    return Counter(s) == Counter(t)

def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = tuple(sorted(s))
        groups.setdefault(key, []).append(s)
    return list(groups.values())`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'palindrome-check',
    title: 'Palindrome Patterns',
    category: 'Strings',
    description: 'Check if string reads same forwards and backwards. Use two pointers from ends moving inward.',
    keyPoints: [
      'Two pointer: O(n) time, O(1) space',
      'Skip non-alphanumeric for sentences',
      'Expand from center for substrings',
      'Used in: Valid Palindrome, Longest Palindromic Substring',
    ],
    codeExample: `def is_palindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

def longest_palindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l, r = l - 1, r + 1
        return s[l+1:r]

    result = ""
    for i in range(len(s)):
        odd = expand(i, i)
        even = expand(i, i + 1)
        result = max(result, odd, even, key=len)
    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'kmp-algorithm',
    title: 'KMP String Matching',
    category: 'Strings',
    description: 'Pattern matching using failure function to avoid redundant comparisons. O(n+m) complexity.',
    keyPoints: [
      'Build LPS (Longest Proper Prefix Suffix) array',
      'Never backtrack in text string',
      'O(n+m) vs O(nm) naive approach',
      'Used in: strStr(), Pattern Matching',
    ],
    codeExample: `def kmp_search(text, pattern):
    def build_lps(p):
        lps = [0] * len(p)
        length = 0
        i = 1
        while i < len(p):
            if p[i] == p[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length:
                length = lps[length - 1]
            else:
                i += 1
        return lps

    lps = build_lps(pattern)
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i, j = i + 1, j + 1
            if j == len(pattern):
                return i - j
        elif j:
            j = lps[j - 1]
        else:
            i += 1
    return -1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'rabin-karp',
    title: 'Rabin-Karp Algorithm',
    category: 'Strings',
    description: 'Rolling hash for pattern matching. Compute hash of pattern and sliding window of text.',
    keyPoints: [
      'O(n+m) average, O(nm) worst case',
      'Use rolling hash to update in O(1)',
      'Good for multiple pattern search',
      'Used in: Repeated DNA Sequences, Plagiarism Detection',
    ],
    codeExample: `def rabin_karp(text, pattern):
    base, mod = 26, 10**9 + 7
    m, n = len(pattern), len(text)

    def hash_val(s):
        h = 0
        for c in s:
            h = (h * base + ord(c)) % mod
        return h

    pattern_hash = hash_val(pattern)
    power = pow(base, m - 1, mod)

    curr_hash = hash_val(text[:m])
    for i in range(n - m + 1):
        if curr_hash == pattern_hash:
            if text[i:i+m] == pattern:
                return i
        if i + m < n:
            curr_hash = ((curr_hash - ord(text[i]) * power) * base + ord(text[i+m])) % mod
    return -1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'string-compression',
    title: 'String Compression',
    category: 'Strings',
    description: 'Compress string using run-length encoding. Count consecutive characters.',
    keyPoints: [
      'Two pointer: read and write pointers',
      'In-place with O(1) extra space',
      'Handle single vs multiple occurrences',
      'Used in: String Compression, RLE Encoding',
    ],
    codeExample: `def compress(chars):
    write = 0
    read = 0

    while read < len(chars):
        char = chars[read]
        count = 0
        while read < len(chars) and chars[read] == char:
            read += 1
            count += 1

        chars[write] = char
        write += 1
        if count > 1:
            for c in str(count):
                chars[write] = c
                write += 1

    return write`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== LINKED LISTS =====
  {
    id: 'fast-slow-pointer',
    title: 'Fast & Slow Pointers',
    category: 'Linked Lists',
    description: 'Two pointers at different speeds. Fast moves 2x, slow 1x. Meet if cycle exists.',
    keyPoints: [
      'Cycle detection (Floyd\'s algorithm)',
      'Find middle of linked list',
      'O(n) time, O(1) space',
      'Used in: Linked List Cycle, Find Middle',
    ],
    codeExample: `def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'reverse-linked-list',
    title: 'Reverse Linked List',
    category: 'Linked Lists',
    description: 'Reverse list by changing next pointers. Track prev, curr, next nodes.',
    keyPoints: [
      'Iterative: O(n) time, O(1) space',
      'Recursive: O(n) time, O(n) stack',
      'Can reverse partial list (m to n)',
      'Used in: Reverse List, Reverse Between',
    ],
    codeExample: `def reverse_list(head):
    prev = None
    curr = head

    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    return prev

def reverse_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'merge-sorted-lists',
    title: 'Merge Sorted Lists',
    category: 'Linked Lists',
    description: 'Merge two sorted lists into one sorted list. Compare heads, advance smaller.',
    keyPoints: [
      'O(n+m) time, O(1) space iterative',
      'Use dummy node for cleaner code',
      'Extends to K lists with heap',
      'Used in: Merge Two Lists, Merge K Lists',
    ],
    codeExample: `def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    curr = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    curr.next = l1 or l2
    return dummy.next`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'remove-nth-from-end',
    title: 'Remove Nth From End',
    category: 'Linked Lists',
    description: 'Remove nth node from end using two pointers n nodes apart.',
    keyPoints: [
      'Two pointers with n gap',
      'Single pass solution',
      'Use dummy for edge cases',
      'O(n) time, O(1) space',
    ],
    codeExample: `def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    slow = fast = dummy

    for _ in range(n + 1):
        fast = fast.next

    while fast:
        slow = slow.next
        fast = fast.next

    slow.next = slow.next.next
    return dummy.next`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'reorder-list',
    title: 'Reorder List',
    category: 'Linked Lists',
    description: 'Reorder L0→L1→...→Ln to L0→Ln→L1→Ln-1. Find middle, reverse second half, merge.',
    keyPoints: [
      'Three steps: find middle, reverse, merge',
      'O(n) time, O(1) space',
      'Combine multiple linked list techniques',
      'Used in: Reorder List, Palindrome Linked List',
    ],
    codeExample: `def reorder_list(head):
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse second half
    prev, curr = None, slow.next
    slow.next = None
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    # Merge two halves
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'lru-cache',
    title: 'LRU Cache',
    category: 'Linked Lists',
    description: 'Least Recently Used cache using hashmap + doubly linked list for O(1) get/put.',
    keyPoints: [
      'HashMap for O(1) key lookup',
      'Doubly linked list for O(1) reorder',
      'Most recent at head, LRU at tail',
      'Used in: LRU Cache, Design problems',
    ],
    codeExample: `class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}  # key -> node
        self.head = self.tail = None

    def get(self, key):
        if key not in self.cache:
            return -1
        self._move_to_front(key)
        return self.cache[key].val

    def put(self, key, val):
        if key in self.cache:
            self.cache[key].val = val
            self._move_to_front(key)
        else:
            if len(self.cache) >= self.cap:
                self._remove_lru()
            self._add_to_front(key, val)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== STACKS & QUEUES =====
  {
    id: 'valid-parentheses',
    title: 'Valid Parentheses',
    category: 'Stacks',
    description: 'Match opening/closing brackets using stack. Push opens, pop and match closes.',
    keyPoints: [
      'Stack for matching pairs',
      'O(n) time, O(n) space',
      'Use hashmap for bracket pairs',
      'Check stack empty at end',
    ],
    codeExample: `def is_valid(s):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
        else:
            stack.append(char)

    return len(stack) == 0`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'monotonic-stack',
    title: 'Monotonic Stack',
    category: 'Stacks',
    description: 'Stack maintaining monotonic order. Pop elements that break the order.',
    keyPoints: [
      'Increasing or decreasing order',
      'O(n) - each element pushed/popped once',
      'Find next greater/smaller element',
      'Used in: Daily Temperatures, Stock Span',
    ],
    codeExample: `def next_greater_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # indices

    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'min-stack',
    title: 'Min Stack',
    category: 'Stacks',
    description: 'Stack with O(1) getMin. Store pairs of (value, currentMin) or use two stacks.',
    keyPoints: [
      'Track minimum at each level',
      'O(1) for all operations',
      'Space: O(n) for min tracking',
      'Used in: Min Stack, Max Stack',
    ],
    codeExample: `class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'evaluate-rpn',
    title: 'Evaluate RPN',
    category: 'Stacks',
    description: 'Evaluate Reverse Polish Notation. Push numbers, pop two for operators.',
    keyPoints: [
      'Numbers: push to stack',
      'Operators: pop 2, compute, push result',
      'O(n) time and space',
      'Used in: RPN Evaluation, Expression Parsing',
    ],
    codeExample: `def eval_rpn(tokens):
    stack = []
    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b)
    }

    for token in tokens:
        if token in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[token](a, b))
        else:
            stack.append(int(token))

    return stack[0]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'sliding-window-max',
    title: 'Sliding Window Maximum',
    category: 'Queues',
    description: 'Find max in each sliding window using monotonic deque.',
    keyPoints: [
      'Deque stores indices in decreasing order',
      'Remove indices outside window',
      'O(n) time - each element added/removed once',
      'Used in: Sliding Window Maximum',
    ],
    codeExample: `from collections import deque

def max_sliding_window(nums, k):
    dq = deque()  # indices
    result = []

    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'queue-using-stacks',
    title: 'Queue Using Stacks',
    category: 'Queues',
    description: 'Implement FIFO queue using two LIFO stacks. Lazy transfer on dequeue.',
    keyPoints: [
      'Two stacks: input and output',
      'Amortized O(1) per operation',
      'Transfer only when output empty',
      'Used in: Implement Queue using Stacks',
    ],
    codeExample: `class MyQueue:
    def __init__(self):
        self.input = []
        self.output = []

    def push(self, x):
        self.input.append(x)

    def pop(self):
        self._transfer()
        return self.output.pop()

    def peek(self):
        self._transfer()
        return self.output[-1]

    def _transfer(self):
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== HASH MAPS =====
  {
    id: 'two-sum-hashmap',
    title: 'Two Sum Pattern',
    category: 'Hash Maps',
    description: 'Find pair summing to target using hashmap to store complements.',
    keyPoints: [
      'O(n) time, O(n) space',
      'Store value -> index mapping',
      'Check complement before adding',
      'Used in: Two Sum, Pair Sum variations',
    ],
    codeExample: `def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'subarray-sum-k',
    title: 'Subarray Sum Equals K',
    category: 'Hash Maps',
    description: 'Count subarrays with sum K using prefix sum + hashmap.',
    keyPoints: [
      'prefix[j] - prefix[i] = k means subarray sum',
      'Store prefix sum frequencies',
      'O(n) time and space',
      'Used in: Subarray Sum Equals K, Path Sum III',
    ],
    codeExample: `def subarray_sum(nums, k):
    count = 0
    prefix_sum = 0
    prefix_counts = {0: 1}

    for num in nums:
        prefix_sum += num
        if prefix_sum - k in prefix_counts:
            count += prefix_counts[prefix_sum - k]
        prefix_counts[prefix_sum] = prefix_counts.get(prefix_sum, 0) + 1

    return count`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'longest-consecutive',
    title: 'Longest Consecutive Sequence',
    category: 'Hash Maps',
    description: 'Find longest consecutive sequence using set for O(1) lookup.',
    keyPoints: [
      'Use set for O(1) lookup',
      'Only start counting from sequence start',
      'O(n) time despite nested loop',
      'Used in: Longest Consecutive Sequence',
    ],
    codeExample: `def longest_consecutive(nums):
    num_set = set(nums)
    max_length = 0

    for num in num_set:
        # Only start from sequence beginning
        if num - 1 not in num_set:
            length = 1
            while num + length in num_set:
                length += 1
            max_length = max(max_length, length)

    return max_length`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'top-k-frequent',
    title: 'Top K Frequent Elements',
    category: 'Hash Maps',
    description: 'Find K most frequent using counting + bucket sort or heap.',
    keyPoints: [
      'Count frequencies with hashmap',
      'Bucket sort: O(n), Heap: O(n log k)',
      'Bucket index = frequency',
      'Used in: Top K Frequent, K Frequent Words',
    ],
    codeExample: `def top_k_frequent(nums, k):
    from collections import Counter

    count = Counter(nums)

    # Bucket sort approach
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)

    result = []
    for i in range(len(buckets) - 1, -1, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== BINARY TREES =====
  {
    id: 'tree-traversal',
    title: 'Tree Traversals',
    category: 'Trees',
    description: 'DFS: preorder (NLR), inorder (LNR), postorder (LRN). BFS: level order.',
    keyPoints: [
      'Preorder: root first, good for copying',
      'Inorder: sorted order for BST',
      'Postorder: children first, good for deletion',
      'BFS: level by level using queue',
    ],
    codeExample: `def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []

def preorder(root):
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []

def level_order(root):
    if not root: return []
    result, queue = [], [root]
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'max-depth',
    title: 'Maximum Depth of Tree',
    category: 'Trees',
    description: 'Find height of tree. Recursively compute max of left and right subtree depths.',
    keyPoints: [
      'Base case: null node returns 0',
      'Return 1 + max(left, right)',
      'O(n) time, O(h) space',
      'Used in: Max Depth, Balanced Tree Check',
    ],
    codeExample: `def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

def min_depth(root):
    if not root:
        return 0
    if not root.left:
        return 1 + min_depth(root.right)
    if not root.right:
        return 1 + min_depth(root.left)
    return 1 + min(min_depth(root.left), min_depth(root.right))`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'same-tree',
    title: 'Same Tree / Symmetric Tree',
    category: 'Trees',
    description: 'Check if trees are identical or mirror images. Compare structure and values.',
    keyPoints: [
      'Same: compare root, left, right',
      'Symmetric: compare left with right mirror',
      'O(n) time, O(h) space',
      'Used in: Same Tree, Symmetric Tree, Subtree',
    ],
    codeExample: `def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

def is_symmetric(root):
    def is_mirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return t1.val == t2.val and is_mirror(t1.left, t2.right) and is_mirror(t1.right, t2.left)
    return is_mirror(root, root)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'lca',
    title: 'Lowest Common Ancestor',
    category: 'Trees',
    description: 'Find LCA of two nodes. Node is LCA if p and q are in different subtrees or one is ancestor.',
    keyPoints: [
      'Return node if it matches p or q',
      'If both subtrees return non-null, current is LCA',
      'O(n) time, O(h) space',
      'BST variant uses value comparison',
    ],
    codeExample: `def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root
    return left or right

def lca_bst(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'path-sum',
    title: 'Path Sum Problems',
    category: 'Trees',
    description: 'Find paths in tree that sum to target. Track current path and sum.',
    keyPoints: [
      'Root-to-leaf: check at leaf nodes',
      'Any path: use prefix sum technique',
      'Return paths: backtrack after exploring',
      'Used in: Path Sum I/II/III, Binary Tree Paths',
    ],
    codeExample: `def has_path_sum(root, target):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target
    return has_path_sum(root.left, target - root.val) or \\
           has_path_sum(root.right, target - root.val)

def path_sum_iii(root, target):
    def dfs(node, prefix_sum, prefix_counts):
        if not node:
            return 0
        prefix_sum += node.val
        count = prefix_counts.get(prefix_sum - target, 0)
        prefix_counts[prefix_sum] = prefix_counts.get(prefix_sum, 0) + 1
        count += dfs(node.left, prefix_sum, prefix_counts)
        count += dfs(node.right, prefix_sum, prefix_counts)
        prefix_counts[prefix_sum] -= 1
        return count
    return dfs(root, 0, {0: 1})`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'serialize-tree',
    title: 'Serialize/Deserialize Tree',
    category: 'Trees',
    description: 'Convert tree to string and back. Use preorder with null markers.',
    keyPoints: [
      'Preorder traversal with null markers',
      'Use delimiter between values',
      'Deserialize recursively',
      'Used in: Serialize Binary Tree, Codec Design',
    ],
    codeExample: `class Codec:
    def serialize(self, root):
        def dfs(node):
            if not node:
                return "null,"
            return str(node.val) + "," + dfs(node.left) + dfs(node.right)
        return dfs(root)

    def deserialize(self, data):
        vals = iter(data.split(","))
        def dfs():
            val = next(vals)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        return dfs()`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'construct-tree',
    title: 'Construct Tree from Traversals',
    category: 'Trees',
    description: 'Build tree from inorder + preorder/postorder. Use recursion with index tracking.',
    keyPoints: [
      'Preorder first = root, find in inorder',
      'Inorder splits left and right subtrees',
      'Use hashmap for O(1) index lookup',
      'O(n) time with hashmap optimization',
    ],
    codeExample: `def build_tree(preorder, inorder):
    inorder_map = {val: i for i, val in enumerate(inorder)}
    pre_idx = [0]

    def build(left, right):
        if left > right:
            return None

        root_val = preorder[pre_idx[0]]
        pre_idx[0] += 1
        root = TreeNode(root_val)

        mid = inorder_map[root_val]
        root.left = build(left, mid - 1)
        root.right = build(mid + 1, right)

        return root

    return build(0, len(inorder) - 1)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== BST =====
  {
    id: 'validate-bst',
    title: 'Validate BST',
    category: 'BST',
    description: 'Check if tree is valid BST. Each node must be within valid range.',
    keyPoints: [
      'Track min/max bounds for each node',
      'Inorder traversal gives sorted order',
      'O(n) time, O(h) space',
      'Used in: Validate BST, Recover BST',
    ],
    codeExample: `def is_valid_bst(root):
    def validate(node, min_val, max_val):
        if not node:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return validate(node.left, min_val, node.val) and \\
               validate(node.right, node.val, max_val)

    return validate(root, float('-inf'), float('inf'))`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'bst-operations',
    title: 'BST Insert/Delete/Search',
    category: 'BST',
    description: 'Standard BST operations. Search/insert are straightforward, delete has 3 cases.',
    keyPoints: [
      'Search: compare and go left/right',
      'Insert: find null position and add',
      'Delete: leaf, one child, two children cases',
      'O(h) average, O(n) worst case',
    ],
    codeExample: `def search_bst(root, val):
    if not root or root.val == val:
        return root
    return search_bst(root.left if val < root.val else root.right, val)

def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    return root

def delete_bst(root, key):
    if not root:
        return None
    if key < root.val:
        root.left = delete_bst(root.left, key)
    elif key > root.val:
        root.right = delete_bst(root.right, key)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        # Find inorder successor
        succ = root.right
        while succ.left:
            succ = succ.left
        root.val = succ.val
        root.right = delete_bst(root.right, succ.val)
    return root`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'kth-smallest-bst',
    title: 'Kth Smallest in BST',
    category: 'BST',
    description: 'Find kth smallest using inorder traversal. BST inorder gives sorted order.',
    keyPoints: [
      'Inorder: left, root, right',
      'Count nodes during traversal',
      'O(H + k) time where H is height',
      'Used in: Kth Smallest, Kth Largest',
    ],
    codeExample: `def kth_smallest(root, k):
    stack = []
    curr = root

    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left

        curr = stack.pop()
        k -= 1
        if k == 0:
            return curr.val

        curr = curr.right`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== HEAPS =====
  {
    id: 'heap-basics',
    title: 'Heap Fundamentals',
    category: 'Heaps',
    description: 'Complete binary tree with heap property. Min-heap: parent <= children. Max-heap: parent >= children.',
    keyPoints: [
      'Insert/Extract: O(log n)',
      'Get min/max: O(1)',
      'Heapify array: O(n)',
      'Python: heapq is min-heap by default',
    ],
    codeExample: `import heapq

# Min heap operations
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
min_val = heapq.heappop(heap)  # 1

# Max heap: negate values
max_heap = []
heapq.heappush(max_heap, -3)
max_val = -heapq.heappop(max_heap)  # 3

# Heapify existing list
arr = [3, 1, 4, 1, 5]
heapq.heapify(arr)  # O(n)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'kth-largest',
    title: 'Kth Largest Element',
    category: 'Heaps',
    description: 'Find kth largest using min-heap of size k. Top of heap is kth largest.',
    keyPoints: [
      'Min-heap of size k',
      'Push all, pop if size > k',
      'O(n log k) time, O(k) space',
      'QuickSelect: O(n) average',
    ],
    codeExample: `import heapq

def find_kth_largest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]

# Alternative: QuickSelect O(n) average
def quick_select(nums, k):
    k = len(nums) - k
    def partition(l, r):
        pivot = nums[r]
        i = l
        for j in range(l, r):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i

    l, r = 0, len(nums) - 1
    while True:
        p = partition(l, r)
        if p == k:
            return nums[p]
        elif p < k:
            l = p + 1
        else:
            r = p - 1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'merge-k-lists',
    title: 'Merge K Sorted Lists',
    category: 'Heaps',
    description: 'Merge k sorted lists using min-heap to always get smallest element.',
    keyPoints: [
      'Push first element of each list',
      'Pop min, push next from same list',
      'O(N log k) where N is total elements',
      'Used in: Merge K Lists, External Sort',
    ],
    codeExample: `import heapq

def merge_k_lists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))

    dummy = curr = ListNode(0)

    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'median-stream',
    title: 'Find Median from Stream',
    category: 'Heaps',
    description: 'Maintain running median using two heaps. Max-heap for lower half, min-heap for upper.',
    keyPoints: [
      'Two heaps balanced in size',
      'Max-heap: lower half, Min-heap: upper half',
      'Median: top of one or average of both',
      'O(log n) add, O(1) median',
    ],
    codeExample: `import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []  # max heap (negated)
        self.hi = []  # min heap

    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))

        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'task-scheduler',
    title: 'Task Scheduler',
    category: 'Heaps',
    description: 'Schedule tasks with cooldown. Use max-heap to always pick most frequent task.',
    keyPoints: [
      'Max-heap by frequency',
      'Track cooldown with queue',
      'Greedy: always pick most frequent available',
      'O(n) time where n is total tasks',
    ],
    codeExample: `from collections import Counter, deque
import heapq

def least_interval(tasks, n):
    count = Counter(tasks)
    max_heap = [-c for c in count.values()]
    heapq.heapify(max_heap)

    time = 0
    cooldown = deque()  # (count, available_time)

    while max_heap or cooldown:
        time += 1

        if max_heap:
            cnt = heapq.heappop(max_heap) + 1
            if cnt:
                cooldown.append((cnt, time + n))

        if cooldown and cooldown[0][1] == time:
            heapq.heappush(max_heap, cooldown.popleft()[0])

    return time`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== GRAPHS =====
  {
    id: 'graph-bfs',
    title: 'Graph BFS',
    category: 'Graphs',
    description: 'Explore graph level by level. Use queue, good for shortest path in unweighted graphs.',
    keyPoints: [
      'Queue-based traversal',
      'Visit neighbors before going deeper',
      'O(V + E) time and space',
      'Used in: Shortest Path, Level Order, Word Ladder',
    ],
    codeExample: `from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result

def shortest_path(graph, start, end):
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'graph-dfs',
    title: 'Graph DFS',
    category: 'Graphs',
    description: 'Explore graph by going deep first. Use recursion or stack.',
    keyPoints: [
      'Go deep before backtracking',
      'Recursive or stack-based',
      'O(V + E) time and space',
      'Used in: Path Finding, Cycle Detection, Connected Components',
    ],
    codeExample: `def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()

    visited.add(node)
    result = [node]

    for neighbor in graph[node]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))

    return result

def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in graph[node]:
                stack.append(neighbor)

    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'topological-sort',
    title: 'Topological Sort',
    category: 'Graphs',
    description: 'Linear ordering of DAG vertices. Every directed edge u→v, u comes before v.',
    keyPoints: [
      'Only valid for DAGs',
      'Kahn\'s: BFS with indegree',
      'DFS: reverse postorder',
      'Used in: Course Schedule, Build Order',
    ],
    codeExample: `from collections import deque

def topological_sort(n, edges):
    graph = {i: [] for i in range(n)}
    indegree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    queue = deque([i for i in range(n) if indegree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == n else []  # Empty if cycle`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'cycle-detection',
    title: 'Cycle Detection',
    category: 'Graphs',
    description: 'Detect cycles in directed/undirected graphs using DFS with color marking.',
    keyPoints: [
      'Directed: white/gray/black coloring',
      'Undirected: track parent node',
      'Gray node revisited = cycle',
      'Used in: Course Schedule, Deadlock Detection',
    ],
    codeExample: `def has_cycle_directed(graph, n):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    return any(color[i] == WHITE and dfs(i) for i in range(n))

def has_cycle_undirected(graph, n):
    visited = [False] * n

    def dfs(node, parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    return any(not visited[i] and dfs(i, -1) for i in range(n))`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'dijkstra',
    title: "Dijkstra's Algorithm",
    category: 'Graphs',
    description: 'Shortest path from source to all vertices in weighted graph. Uses priority queue.',
    keyPoints: [
      'Greedy: always process closest unvisited',
      'Priority queue for efficiency',
      'O((V + E) log V) with heap',
      'Only works with non-negative weights',
    ],
    codeExample: `import heapq

def dijkstra(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    return dist`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'bellman-ford',
    title: 'Bellman-Ford Algorithm',
    category: 'Graphs',
    description: 'Shortest path with negative weights. Relax all edges V-1 times.',
    keyPoints: [
      'Works with negative weights',
      'Detects negative cycles',
      'O(VE) time complexity',
      'Used in: Cheapest Flights with K Stops',
    ],
    codeExample: `def bellman_ford(n, edges, start):
    dist = [float('inf')] * n
    dist[start] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Check for negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None  # Negative cycle exists

    return dist`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'union-find',
    title: 'Union-Find (DSU)',
    category: 'Graphs',
    description: 'Disjoint Set Union with path compression and union by rank. Near O(1) operations.',
    keyPoints: [
      'Find: path compression to root',
      'Union: by rank/size',
      'O(α(n)) ≈ O(1) amortized',
      'Used in: Connected Components, Kruskal\'s MST',
    ],
    codeExample: `class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'mst-kruskal',
    title: "Kruskal's MST",
    category: 'Graphs',
    description: 'Build minimum spanning tree by adding edges in weight order if no cycle.',
    keyPoints: [
      'Sort edges by weight',
      'Use Union-Find for cycle detection',
      'O(E log E) time',
      'Used in: Min Cost to Connect Points, Network Connection',
    ],
    codeExample: `def kruskal(n, edges):
    edges.sort(key=lambda x: x[2])  # Sort by weight
    uf = UnionFind(n)
    mst = []
    cost = 0

    for u, v, w in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            cost += w
            if len(mst) == n - 1:
                break

    return cost if len(mst) == n - 1 else -1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'bipartite',
    title: 'Bipartite Graph Check',
    category: 'Graphs',
    description: 'Check if graph can be 2-colored. Use BFS/DFS and alternate colors.',
    keyPoints: [
      'Two-color graph alternating colors',
      'BFS or DFS with coloring',
      'O(V + E) time',
      'Used in: Is Graph Bipartite, Possible Bipartition',
    ],
    codeExample: `from collections import deque

def is_bipartite(graph):
    n = len(graph)
    color = [-1] * n

    for start in range(n):
        if color[start] != -1:
            continue

        queue = deque([start])
        color[start] = 0

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False

    return True`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'number-of-islands',
    title: 'Number of Islands',
    category: 'Graphs',
    description: 'Count connected components in a grid. Use BFS/DFS to mark visited cells.',
    keyPoints: [
      'Grid as implicit graph',
      '4-directional or 8-directional neighbors',
      'Mark visited by changing cell value',
      'Used in: Number of Islands, Max Area of Island',
    ],
    codeExample: `def num_islands(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'  # Mark visited
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1

    return count`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== BINARY SEARCH =====
  {
    id: 'binary-search-basic',
    title: 'Binary Search',
    category: 'Binary Search',
    description: 'Divide and conquer on sorted data. Eliminate half each iteration.',
    keyPoints: [
      'Requires sorted input',
      'O(log n) time complexity',
      'mid = left + (right - left) // 2',
      'Used in: Search, Insert Position',
    ],
    codeExample: `def binary_search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'lower-upper-bound',
    title: 'Lower/Upper Bound',
    category: 'Binary Search',
    description: 'Find first/last position of target. Use different conditions for bound type.',
    keyPoints: [
      'Lower: first index where arr[i] >= target',
      'Upper: first index where arr[i] > target',
      'Useful for range queries',
      'Used in: First and Last Position, Count Occurrences',
    ],
    codeExample: `def lower_bound(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

def upper_bound(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'search-rotated',
    title: 'Search in Rotated Array',
    category: 'Binary Search',
    description: 'Binary search in rotated sorted array. Determine which half is sorted.',
    keyPoints: [
      'One half is always sorted',
      'Check if target in sorted half',
      'O(log n) time',
      'Used in: Search Rotated Array, Find Minimum',
    ],
    codeExample: `def search_rotated(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'binary-search-answer',
    title: 'Binary Search on Answer',
    category: 'Binary Search',
    description: 'Search for optimal value in a range. Check if value satisfies condition.',
    keyPoints: [
      'Search space is answer range, not array',
      'Feasibility check function',
      'O(log(range) * check_cost)',
      'Used in: Koko Eating Bananas, Split Array Largest Sum',
    ],
    codeExample: `def min_eating_speed(piles, h):
    def can_finish(speed):
        hours = sum((p + speed - 1) // speed for p in piles)
        return hours <= h

    left, right = 1, max(piles)

    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1

    return left`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'find-peak',
    title: 'Find Peak Element',
    category: 'Binary Search',
    description: 'Find local maximum in array. Move towards ascending direction.',
    keyPoints: [
      'Peak: element greater than neighbors',
      'Always move towards higher neighbor',
      'O(log n) time',
      'Used in: Find Peak, Mountain Array',
    ],
    codeExample: `def find_peak(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid

    return left`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== DYNAMIC PROGRAMMING =====
  {
    id: 'dp-basics',
    title: 'DP Fundamentals',
    category: 'Dynamic Programming',
    description: 'Break into subproblems, store solutions. Bottom-up or top-down with memoization.',
    keyPoints: [
      'Optimal substructure + overlapping subproblems',
      'Define state and recurrence',
      'Memoization (top-down) or tabulation (bottom-up)',
      'Often can optimize space',
    ],
    codeExample: `# Top-down with memoization
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1) + fib_memo(n-2)
    return memo[n]

# Bottom-up tabulation
def fib_tab(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Space optimized
def fib_opt(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'climbing-stairs',
    title: 'Climbing Stairs',
    category: 'Dynamic Programming',
    description: 'Count ways to reach top. dp[i] = dp[i-1] + dp[i-2] (take 1 or 2 steps).',
    keyPoints: [
      'Base case: dp[0] = 1, dp[1] = 1',
      'Recurrence: dp[i] = dp[i-1] + dp[i-2]',
      'Same as Fibonacci',
      'Extends to k steps: sum of last k values',
    ],
    codeExample: `def climb_stairs(n):
    if n <= 2:
        return n
    prev, curr = 1, 2
    for _ in range(3, n + 1):
        prev, curr = curr, prev + curr
    return curr

# K steps variant
def climb_k_stairs(n, k):
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i] += dp[i - j]
    return dp[n]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'coin-change',
    title: 'Coin Change',
    category: 'Dynamic Programming',
    description: 'Minimum coins for amount. dp[i] = min coins for amount i.',
    keyPoints: [
      'Unbounded knapsack variant',
      'dp[i] = min(dp[i], dp[i-coin] + 1)',
      'Initialize with infinity',
      'O(amount * coins) time',
    ],
    codeExample: `def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# Count number of ways
def coin_change_ways(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'knapsack',
    title: '0/1 Knapsack',
    category: 'Dynamic Programming',
    description: 'Max value with weight limit. Each item used at most once.',
    keyPoints: [
      'dp[i][w] = max value with first i items, capacity w',
      'Take or skip each item',
      'O(nW) time and space, can optimize to O(W)',
      'Used in: Partition Equal Subset Sum, Target Sum',
    ],
    codeExample: `def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]  # Don't take
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                    dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][capacity]

# Space optimized (iterate backwards)
def knapsack_opt(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'lcs',
    title: 'Longest Common Subsequence',
    category: 'Dynamic Programming',
    description: 'Find length of LCS of two strings. Classic 2D DP problem.',
    keyPoints: [
      'dp[i][j] = LCS of first i chars of s1, first j of s2',
      'If match: dp[i-1][j-1] + 1',
      'Else: max(dp[i-1][j], dp[i][j-1])',
      'O(mn) time, can optimize to O(min(m,n)) space',
    ],
    codeExample: `def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'lis',
    title: 'Longest Increasing Subsequence',
    category: 'Dynamic Programming',
    description: 'Find length of LIS. O(n²) DP or O(n log n) with binary search.',
    keyPoints: [
      'dp[i] = LIS ending at index i',
      'Binary search: maintain sorted tail array',
      'O(n log n) optimal solution',
      'Used in: LIS, Russian Doll Envelopes',
    ],
    codeExample: `def length_of_lis(nums):
    # O(n^2) DP
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def lis_binary_search(nums):
    # O(n log n) with binary search
    tails = []
    for num in nums:
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    return len(tails)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'edit-distance',
    title: 'Edit Distance',
    category: 'Dynamic Programming',
    description: 'Min operations (insert, delete, replace) to convert s1 to s2.',
    keyPoints: [
      'dp[i][j] = edit distance for first i, j chars',
      'Match: dp[i-1][j-1]',
      'Else: 1 + min(insert, delete, replace)',
      'O(mn) time and space',
    ],
    codeExample: `def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # delete
                    dp[i][j-1],     # insert
                    dp[i-1][j-1]    # replace
                )

    return dp[m][n]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'word-break',
    title: 'Word Break',
    category: 'Dynamic Programming',
    description: 'Check if string can be segmented into dictionary words.',
    keyPoints: [
      'dp[i] = can segment first i characters',
      'Check all possible last words',
      'O(n²) time with set lookup',
      'Used in: Word Break I/II',
    ],
    codeExample: `def word_break(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'house-robber',
    title: 'House Robber',
    category: 'Dynamic Programming',
    description: 'Max sum without adjacent elements. dp[i] = max(take, skip).',
    keyPoints: [
      'dp[i] = max money robbing up to house i',
      'Take: dp[i-2] + nums[i]',
      'Skip: dp[i-1]',
      'O(n) time, O(1) space',
    ],
    codeExample: `def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev2, prev1 = 0, 0
    for num in nums:
        curr = max(prev2 + num, prev1)
        prev2, prev1 = prev1, curr

    return prev1

# Circular houses
def rob_circular(nums):
    if len(nums) == 1:
        return nums[0]
    return max(rob(nums[:-1]), rob(nums[1:]))`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'unique-paths',
    title: 'Unique Paths',
    category: 'Dynamic Programming',
    description: 'Count paths in grid from top-left to bottom-right, moving only right or down.',
    keyPoints: [
      'dp[i][j] = paths to reach (i,j)',
      'dp[i][j] = dp[i-1][j] + dp[i][j-1]',
      'Can optimize to O(n) space',
      'With obstacles: set blocked cells to 0',
    ],
    codeExample: `def unique_paths(m, n):
    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]

    return dp[n-1]

def unique_paths_obstacles(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n
    dp[0] = 1

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]

    return dp[n-1]`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'stock-problems',
    title: 'Best Time to Buy/Sell Stock',
    category: 'Dynamic Programming',
    description: 'Maximize profit from stock transactions. Track states based on constraints.',
    keyPoints: [
      'One transaction: track min price, max profit',
      'Multiple: sum all positive differences',
      'With cooldown: state machine DP',
      'K transactions: dp[k][i] states',
    ],
    codeExample: `# One transaction
def max_profit_one(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit

# Unlimited transactions
def max_profit_unlimited(prices):
    return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))

# With cooldown
def max_profit_cooldown(prices):
    if not prices:
        return 0
    hold, sold, rest = -prices[0], 0, 0
    for price in prices[1:]:
        hold, sold, rest = max(hold, rest - price), hold + price, max(rest, sold)
    return max(sold, rest)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== BACKTRACKING =====
  {
    id: 'backtracking-template',
    title: 'Backtracking Template',
    category: 'Backtracking',
    description: 'Explore all possibilities, backtrack when constraint violated. Build solution incrementally.',
    keyPoints: [
      'Base case: found solution or dead end',
      'Make choice, recurse, undo choice',
      'Prune invalid paths early',
      'Used in: Permutations, Subsets, N-Queens',
    ],
    codeExample: `def backtrack(path, choices):
    if is_solution(path):
        result.append(path[:])  # Copy path
        return

    for choice in choices:
        if is_valid(choice):
            path.append(choice)       # Make choice
            backtrack(path, updated_choices)
            path.pop()                # Undo choice`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'permutations',
    title: 'Permutations',
    category: 'Backtracking',
    description: 'Generate all orderings of elements. Track used elements.',
    keyPoints: [
      'n! permutations for n elements',
      'Use set/array for used tracking',
      'Handle duplicates with sorting + skip',
      'O(n! * n) time',
    ],
    codeExample: `def permute(nums):
    result = []

    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False

    backtrack([], [False] * len(nums))
    return result

# With duplicates
def permute_unique(nums):
    nums.sort()
    result = []
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i-1] and not used[i-1]):
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'subsets',
    title: 'Subsets',
    category: 'Backtracking',
    description: 'Generate all subsets (power set). Include or exclude each element.',
    keyPoints: [
      '2^n subsets for n elements',
      'Each element: include or exclude',
      'Use index to avoid duplicates',
      'Iterative or recursive approach',
    ],
    codeExample: `def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# With duplicates
def subsets_with_dup(nums):
    nums.sort()
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'combinations',
    title: 'Combinations',
    category: 'Backtracking',
    description: 'Generate all k-size subsets. Similar to subsets but fixed size.',
    keyPoints: [
      'C(n,k) = n! / (k!(n-k)!) combinations',
      'Build path until size k',
      'Prune if remaining elements insufficient',
      'Used in: Combination Sum, Letter Combinations',
    ],
    codeExample: `def combine(n, k):
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        # Prune: need k-len(path) more, have n-i+1 left
        for i in range(start, n - (k - len(path)) + 2):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'combination-sum',
    title: 'Combination Sum',
    category: 'Backtracking',
    description: 'Find combinations that sum to target. Can reuse elements.',
    keyPoints: [
      'Unbounded: same element multiple times',
      'Bounded: each element once',
      'Track remaining target',
      'Sort and prune when sum exceeds',
    ],
    codeExample: `def combination_sum(candidates, target):
    result = []

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # i not i+1 for reuse
            path.pop()

    candidates.sort()
    backtrack(0, [], target)
    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'n-queens',
    title: 'N-Queens',
    category: 'Backtracking',
    description: 'Place N queens on NxN board such that no two attack each other.',
    keyPoints: [
      'One queen per row, find valid column',
      'Track columns, diagonals, anti-diagonals',
      'Diagonal: row - col constant',
      'Anti-diagonal: row + col constant',
    ],
    codeExample: `def solve_n_queens(n):
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    cols = set()
    diag = set()      # row - col
    anti_diag = set() # row + col

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return

        for col in range(n):
            if col in cols or row - col in diag or row + col in anti_diag:
                continue

            cols.add(col)
            diag.add(row - col)
            anti_diag.add(row + col)
            board[row][col] = 'Q'

            backtrack(row + 1)

            cols.remove(col)
            diag.remove(row - col)
            anti_diag.remove(row + col)
            board[row][col] = '.'

    backtrack(0)
    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'word-search',
    title: 'Word Search',
    category: 'Backtracking',
    description: 'Find word in grid by traversing adjacent cells. Mark visited during search.',
    keyPoints: [
      'DFS from each cell matching first char',
      'Mark cell visited during path',
      'Restore cell after backtracking',
      'O(m*n*4^L) where L is word length',
    ],
    codeExample: `def exist(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, i):
        if i == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[i]:
            return False

        temp = board[r][c]
        board[r][c] = '#'  # Mark visited

        found = (backtrack(r+1, c, i+1) or
                 backtrack(r-1, c, i+1) or
                 backtrack(r, c+1, i+1) or
                 backtrack(r, c-1, i+1))

        board[r][c] = temp  # Restore
        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== GREEDY =====
  {
    id: 'greedy-basics',
    title: 'Greedy Algorithms',
    category: 'Greedy',
    description: 'Make locally optimal choice at each step hoping for global optimum.',
    keyPoints: [
      'Works when local optimal leads to global',
      'No backtracking needed',
      'Often requires proof of correctness',
      'Usually more efficient than DP',
    ],
    codeExample: `# Activity Selection
def max_activities(start, end):
    activities = sorted(zip(start, end), key=lambda x: x[1])
    count = 1
    last_end = activities[0][1]

    for s, e in activities[1:]:
        if s >= last_end:
            count += 1
            last_end = e

    return count`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'jump-game',
    title: 'Jump Game',
    category: 'Greedy',
    description: 'Determine if you can reach the end. Track maximum reachable index.',
    keyPoints: [
      'Track furthest reachable position',
      'If current > furthest, cannot proceed',
      'O(n) time, O(1) space',
      'Used in: Jump Game I/II',
    ],
    codeExample: `def can_jump(nums):
    furthest = 0
    for i in range(len(nums)):
        if i > furthest:
            return False
        furthest = max(furthest, i + nums[i])
    return True

def min_jumps(nums):
    jumps = 0
    current_end = 0
    furthest = 0

    for i in range(len(nums) - 1):
        furthest = max(furthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = furthest

    return jumps`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'gas-station',
    title: 'Gas Station',
    category: 'Greedy',
    description: 'Find starting station to complete circular trip. Track total and current tank.',
    keyPoints: [
      'If total gas >= total cost, solution exists',
      'Reset start when tank goes negative',
      'O(n) time, O(1) space',
      'Unique solution guaranteed if exists',
    ],
    codeExample: `def can_complete_circuit(gas, cost):
    total_tank = 0
    curr_tank = 0
    start = 0

    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total_tank += diff
        curr_tank += diff

        if curr_tank < 0:
            start = i + 1
            curr_tank = 0

    return start if total_tank >= 0 else -1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'partition-labels',
    title: 'Partition Labels',
    category: 'Greedy',
    description: 'Partition string so each letter appears in at most one part.',
    keyPoints: [
      'Find last occurrence of each char',
      'Extend partition to include all instances',
      'O(n) time, O(1) space (26 chars)',
      'Used in: Partition Labels',
    ],
    codeExample: `def partition_labels(s):
    last = {c: i for i, c in enumerate(s)}
    result = []
    start = end = 0

    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = i + 1

    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== BIT MANIPULATION =====
  {
    id: 'bit-basics',
    title: 'Bit Manipulation Basics',
    category: 'Bit Manipulation',
    description: 'Operations on individual bits. AND, OR, XOR, NOT, shifts.',
    keyPoints: [
      'AND (&): both 1 → 1',
      'OR (|): either 1 → 1',
      'XOR (^): different → 1, same → 0',
      'Shifts: << multiply by 2, >> divide by 2',
    ],
    codeExample: `# Common bit operations
def get_bit(n, i):
    return (n >> i) & 1

def set_bit(n, i):
    return n | (1 << i)

def clear_bit(n, i):
    return n & ~(1 << i)

def toggle_bit(n, i):
    return n ^ (1 << i)

def count_bits(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# Brian Kernighan's algorithm
def count_bits_fast(n):
    count = 0
    while n:
        n &= n - 1  # Clear lowest set bit
        count += 1
    return count`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'single-number',
    title: 'Single Number',
    category: 'Bit Manipulation',
    description: 'Find element appearing once when others appear twice. XOR all elements.',
    keyPoints: [
      'a ^ a = 0, a ^ 0 = a',
      'XOR is commutative and associative',
      'Pairs cancel out',
      'Extends to thrice with bit counting',
    ],
    codeExample: `def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

# Every element appears 3 times except one
def single_number_ii(nums):
    ones = twos = 0
    for num in nums:
        ones = (ones ^ num) & ~twos
        twos = (twos ^ num) & ~ones
    return ones`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'power-of-two',
    title: 'Power of Two',
    category: 'Bit Manipulation',
    description: 'Check if n is power of 2. Powers of 2 have exactly one set bit.',
    keyPoints: [
      'n & (n-1) clears lowest set bit',
      'Power of 2: only one bit set',
      'n & (n-1) == 0 for power of 2',
      'Handle n <= 0 edge case',
    ],
    codeExample: `def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def is_power_of_four(n):
    # Power of 2 and bit at odd position
    return n > 0 and (n & (n - 1)) == 0 and (n & 0x55555555)

def next_power_of_two(n):
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'missing-number',
    title: 'Missing Number',
    category: 'Bit Manipulation',
    description: 'Find missing number in [0, n]. Use XOR with indices and values.',
    keyPoints: [
      'XOR index with value',
      'All pairs cancel except missing',
      'Also: sum formula n*(n+1)/2',
      'O(n) time, O(1) space',
    ],
    codeExample: `def missing_number(nums):
    result = len(nums)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result

# Alternative: math approach
def missing_number_math(nums):
    n = len(nums)
    expected = n * (n + 1) // 2
    return expected - sum(nums)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== TRIES =====
  {
    id: 'trie-implementation',
    title: 'Trie Implementation',
    category: 'Tries',
    description: 'Tree for storing strings. Each node represents a character.',
    keyPoints: [
      'O(m) insert/search where m is word length',
      'Efficient prefix matching',
      'Space: O(alphabet_size * key_length * n)',
      'Used in: Autocomplete, Spell Check',
    ],
    codeExample: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._find_node(prefix) is not None

    def _find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'word-search-ii',
    title: 'Word Search II',
    category: 'Tries',
    description: 'Find all words from dictionary in grid. Build trie and DFS.',
    keyPoints: [
      'Build trie from word list',
      'DFS from each cell with trie traversal',
      'Prune trie branches after finding word',
      'O(m*n*4^L) where L is max word length',
    ],
    codeExample: `def find_words(board, words):
    trie = Trie()
    for word in words:
        trie.insert(word)

    result = set()
    rows, cols = len(board), len(board[0])

    def dfs(r, c, node, path):
        char = board[r][c]
        if char not in node.children:
            return

        node = node.children[char]
        path += char

        if node.is_end:
            result.add(path)

        board[r][c] = '#'
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                dfs(nr, nc, node, path)
        board[r][c] = char

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root, "")

    return list(result)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== INTERVALS =====
  {
    id: 'interval-operations',
    title: 'Interval Problems',
    category: 'Intervals',
    description: 'Problems involving ranges. Usually require sorting by start or end time.',
    keyPoints: [
      'Sort by start or end time',
      'Check overlap: a.end >= b.start',
      'Track active intervals with heap',
      'Used in: Meeting Rooms, Insert Interval',
    ],
    codeExample: `# Can attend all meetings?
def can_attend(intervals):
    intervals.sort()
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    return True

# Minimum meeting rooms
def min_meeting_rooms(intervals):
    import heapq
    if not intervals:
        return 0

    intervals.sort()
    heap = [intervals[0][1]]  # End times

    for start, end in intervals[1:]:
        if start >= heap[0]:
            heapq.heappop(heap)
        heapq.heappush(heap, end)

    return len(heap)`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== MATH =====
  {
    id: 'gcd-lcm',
    title: 'GCD and LCM',
    category: 'Math',
    description: 'Greatest Common Divisor using Euclidean algorithm. LCM = a*b/GCD.',
    keyPoints: [
      'GCD(a, b) = GCD(b, a % b)',
      'LCM(a, b) = a * b / GCD(a, b)',
      'Extended Euclidean: find x, y where ax + by = gcd',
      'O(log(min(a,b))) time',
    ],
    codeExample: `def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

# Extended Euclidean
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'prime-sieve',
    title: 'Prime Numbers / Sieve',
    category: 'Math',
    description: 'Generate primes using Sieve of Eratosthenes. Mark multiples as composite.',
    keyPoints: [
      'Sieve: O(n log log n) time',
      'Mark multiples starting from p*p',
      'Primality test: check up to sqrt(n)',
      'Used in: Count Primes, Prime Factorization',
    ],
    codeExample: `def sieve_of_eratosthenes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(n + 1) if is_prime[i]]

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
  {
    id: 'modular-arithmetic',
    title: 'Modular Arithmetic',
    category: 'Math',
    description: 'Operations under modulo. Used for large number computations.',
    keyPoints: [
      '(a + b) % m = ((a % m) + (b % m)) % m',
      'Modular exponentiation: O(log n)',
      'Modular inverse: pow(a, m-2, m) when m is prime',
      'Used in: Large Factorial, Combination Counting',
    ],
    codeExample: `def mod_pow(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result

def mod_inverse(a, mod):
    return mod_pow(a, mod - 2, mod)

# nCr mod p
def nCr_mod(n, r, mod):
    if r > n:
        return 0
    num = den = 1
    for i in range(r):
        num = (num * (n - i)) % mod
        den = (den * (i + 1)) % mod
    return (num * mod_inverse(den, mod)) % mod`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },

  // ===== SORTING =====
  {
    id: 'sorting-algorithms',
    title: 'Sorting Algorithms',
    category: 'Sorting',
    description: 'Common sorting algorithms and their characteristics.',
    keyPoints: [
      'QuickSort: O(n log n) avg, O(n²) worst',
      'MergeSort: O(n log n), stable, O(n) space',
      'HeapSort: O(n log n), in-place',
      'Counting/Radix: O(n) for bounded integers',
    ],
    codeExample: `def quicksort(arr, lo=0, hi=None):
    if hi is None:
        hi = len(arr) - 1
    if lo >= hi:
        return

    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]

    quicksort(arr, lo, i - 1)
    quicksort(arr, i + 1, hi)

def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result`,
    codeLanguage: 'python',
    topicType: 'dsa',
  },
];
