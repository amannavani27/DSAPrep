import { Topic } from '../types';

export const systemDesignTopics: Topic[] = [
  // ===== SCALABILITY FUNDAMENTALS =====
  {
    id: 'sd-horizontal-vs-vertical-scaling',
    title: 'Horizontal vs Vertical Scaling',
    category: 'Scalability',
    description: 'Understand the two primary approaches to scaling systems: adding more machines (horizontal) vs upgrading existing machines (vertical).',
    keyPoints: [
      'Vertical: Add more CPU, RAM, storage to existing server',
      'Horizontal: Add more servers to distribute load',
      'Vertical has hardware limits, horizontal is theoretically unlimited',
      'Horizontal requires load balancing and distributed systems design',
      'Most large-scale systems use horizontal scaling',
    ],
    codeExample: `# Vertical Scaling Example
Single Server:
  - Before: 4 CPU, 16GB RAM
  - After: 16 CPU, 64GB RAM

# Horizontal Scaling Example
Load Balancer
    |
-----------
|    |    |
S1   S2   S3  (Multiple identical servers)`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-load-balancing',
    title: 'Load Balancing',
    category: 'Scalability',
    description: 'Distribute incoming traffic across multiple servers to ensure no single server bears too much load. Critical for high availability and scalability.',
    keyPoints: [
      'Algorithms: Round Robin, Least Connections, IP Hash, Weighted',
      'L4 (Transport) vs L7 (Application) load balancers',
      'Health checks to detect and remove unhealthy servers',
      'Session persistence/sticky sessions when needed',
      'Popular: NGINX, HAProxy, AWS ELB, Google Cloud LB',
    ],
    codeExample: `# NGINX Load Balancer Config
upstream backend {
    least_conn;  # Least connections algorithm
    server backend1.example.com weight=5;
    server backend2.example.com;
    server backend3.example.com backup;
}

server {
    location / {
        proxy_pass http://backend;
    }
}`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-caching-strategies',
    title: 'Caching Strategies',
    category: 'Scalability',
    description: 'Store frequently accessed data in fast storage to reduce latency and database load. Choose the right caching strategy based on your use case.',
    keyPoints: [
      'Cache-Aside: App checks cache first, then DB',
      'Write-Through: Write to cache and DB simultaneously',
      'Write-Behind: Write to cache, async write to DB',
      'Read-Through: Cache loads from DB on miss',
      'TTL (Time To Live) for cache expiration',
    ],
    codeExample: `# Cache-Aside Pattern (Python/Redis)
def get_user(user_id):
    # Check cache first
    cached = redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)

    # Cache miss - query database
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)

    # Store in cache with TTL
    redis.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-cdn',
    title: 'Content Delivery Network (CDN)',
    category: 'Scalability',
    description: 'Geographically distributed network of servers that cache and serve content to users from the nearest location, reducing latency.',
    keyPoints: [
      'Edge servers cache static content close to users',
      'Reduces origin server load and bandwidth costs',
      'Types: Pull CDN (lazy) vs Push CDN (proactive)',
      'Best for static assets: images, CSS, JS, videos',
      'Popular: CloudFlare, Akamai, AWS CloudFront, Fastly',
    ],
    codeExample: `# CDN Architecture
User (NYC) --> Edge Server (NYC) --> Cache HIT --> Return Content

User (NYC) --> Edge Server (NYC) --> Cache MISS
                    |
                    v
              Origin Server (SF) --> Return + Cache

# CloudFront Configuration
{
  "Origins": [{
    "DomainName": "my-bucket.s3.amazonaws.com",
    "S3OriginConfig": { "OriginAccessIdentity": "" }
  }],
  "DefaultCacheBehavior": {
    "TTL": 86400,  // 24 hours
    "Compress": true
  }
}`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-database-replication',
    title: 'Database Replication',
    category: 'Scalability',
    description: 'Maintain copies of data across multiple database servers for improved read performance, fault tolerance, and geographic distribution.',
    keyPoints: [
      'Master-Slave: One write node, multiple read replicas',
      'Master-Master: Multiple write nodes (conflict resolution needed)',
      'Synchronous: Strong consistency, higher latency',
      'Asynchronous: Lower latency, eventual consistency',
      'Read replicas handle read-heavy workloads',
    ],
    codeExample: `# Master-Slave Replication Setup
         [Master]
        /   |   \\
       /    |    \\
   [Slave] [Slave] [Slave]

# Writes go to Master
INSERT INTO orders (user_id, total) VALUES (1, 99.99);

# Reads can go to any Slave (with replication lag)
SELECT * FROM orders WHERE user_id = 1;

# Connection routing example
def get_db_connection(operation):
    if operation == 'write':
        return master_connection
    else:
        return random.choice(slave_connections)`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-database-sharding',
    title: 'Database Sharding',
    category: 'Scalability',
    description: 'Horizontally partition data across multiple databases. Each shard holds a subset of data, allowing systems to scale beyond single database limits.',
    keyPoints: [
      'Range-based: Partition by value ranges (e.g., user IDs 1-1M)',
      'Hash-based: Hash key to determine shard (consistent hashing)',
      'Directory-based: Lookup table maps keys to shards',
      'Challenges: Cross-shard queries, rebalancing, hotspots',
      'Shard key selection is critical for even distribution',
    ],
    codeExample: `# Hash-based Sharding
def get_shard(user_id, num_shards=4):
    return hash(user_id) % num_shards

# Example distribution
user_id=123 -> shard_1
user_id=456 -> shard_0
user_id=789 -> shard_3

# Consistent Hashing (better for adding/removing shards)
class ConsistentHash:
    def __init__(self, nodes, virtual_nodes=100):
        self.ring = {}
        for node in nodes:
            for i in range(virtual_nodes):
                key = hash(f"{node}:{i}")
                self.ring[key] = node
        self.sorted_keys = sorted(self.ring.keys())`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },

  // ===== DATABASES =====
  {
    id: 'sd-sql-vs-nosql',
    title: 'SQL vs NoSQL Databases',
    category: 'Databases',
    description: 'Choose the right database type based on your data model, consistency requirements, and scalability needs.',
    keyPoints: [
      'SQL: Structured data, ACID transactions, complex queries',
      'NoSQL: Flexible schema, horizontal scaling, high throughput',
      'Document DBs (MongoDB): JSON-like, nested data',
      'Key-Value (Redis, DynamoDB): Simple, fast lookups',
      'Wide-Column (Cassandra): Time-series, high write volume',
    ],
    codeExample: `# SQL - Relational Data with Joins
SELECT u.name, o.total
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.created_at > '2024-01-01';

# NoSQL (MongoDB) - Embedded Documents
{
  "_id": "user123",
  "name": "John",
  "orders": [
    { "id": "o1", "total": 99.99, "created_at": "2024-01-15" },
    { "id": "o2", "total": 149.99, "created_at": "2024-02-20" }
  ]
}

# When to use each:
SQL: Banking, e-commerce, ERP (consistency critical)
NoSQL: Social feeds, IoT, real-time analytics (scale critical)`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-acid-vs-base',
    title: 'ACID vs BASE',
    category: 'Databases',
    description: 'Two different approaches to database consistency. ACID prioritizes consistency while BASE prioritizes availability.',
    keyPoints: [
      'ACID: Atomicity, Consistency, Isolation, Durability',
      'BASE: Basically Available, Soft state, Eventually consistent',
      'ACID guarantees transactions complete fully or not at all',
      'BASE accepts temporary inconsistency for availability',
      'CAP theorem influences which approach to choose',
    ],
    codeExample: `# ACID Transaction Example
BEGIN TRANSACTION;
  UPDATE accounts SET balance = balance - 100 WHERE id = 1;
  UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;  -- Both succeed or both fail

# BASE Example (Eventually Consistent)
# Write to primary
redis.set("user:123:balance", 500)

# Replicas may show old value temporarily
replica1.get("user:123:balance")  # might be 400 briefly
# Eventually all replicas converge to 500

# CAP Theorem: Pick 2 of 3
Consistency | Availability | Partition Tolerance
     CA     |     (Traditional RDBMS - not partition tolerant)
     CP     |     (MongoDB, HBase - sacrifice availability)
     AP     |     (Cassandra, DynamoDB - sacrifice consistency)`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-database-indexing',
    title: 'Database Indexing',
    category: 'Databases',
    description: 'Create data structures to speed up query performance. Essential for optimizing read-heavy workloads.',
    keyPoints: [
      'B-Tree indexes: Default for most DBs, good for range queries',
      'Hash indexes: O(1) lookups, equality queries only',
      'Composite indexes: Multiple columns, order matters',
      'Covering indexes: Include all query columns to avoid table lookup',
      'Trade-off: Faster reads but slower writes (index maintenance)',
    ],
    codeExample: `# Creating Indexes
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_user_date ON orders(user_id, created_at);

# Query using index
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
-- Uses idx_user_email (index scan instead of full table scan)

# Composite Index Order Matters!
INDEX(a, b, c) can be used for:
  WHERE a = 1
  WHERE a = 1 AND b = 2
  WHERE a = 1 AND b = 2 AND c = 3

Cannot efficiently use for:
  WHERE b = 2 (leftmost column missing)
  WHERE a = 1 AND c = 3 (gap in columns)`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-cap-theorem',
    title: 'CAP Theorem',
    category: 'Databases',
    description: 'In a distributed system, you can only guarantee two of three: Consistency, Availability, and Partition Tolerance.',
    keyPoints: [
      'Consistency: All nodes see same data at same time',
      'Availability: Every request gets a response',
      'Partition Tolerance: System works despite network failures',
      'Networks always fail, so choose between C and A',
      'Real systems are on a spectrum, not binary choices',
    ],
    codeExample: `# CAP Trade-offs in Practice

# CP System (Consistency + Partition Tolerance)
# Example: MongoDB, HBase, Zookeeper
- Refuses to respond if cannot guarantee consistency
- Banking: "Sorry, please try again" vs showing wrong balance

# AP System (Availability + Partition Tolerance)
# Example: Cassandra, DynamoDB, CouchDB
- Always responds, might be stale data
- Social media: Show slightly old likes count vs error

# Partition Scenario
Network splits between Data Center A and Data Center B:

CP Choice:
  - Only one DC accepts writes
  - Other DC returns errors until healed

AP Choice:
  - Both DCs accept writes
  - Conflict resolution needed when healed`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-database-normalization',
    title: 'Database Normalization & Denormalization',
    category: 'Databases',
    description: 'Normalization reduces redundancy; denormalization improves read performance. Know when to apply each.',
    keyPoints: [
      '1NF: Atomic values, no repeating groups',
      '2NF: No partial dependencies on composite keys',
      '3NF: No transitive dependencies',
      'Denormalize for read-heavy workloads (data duplication)',
      'Trade-off: Storage vs query performance',
    ],
    codeExample: `# Normalized (3NF) - Good for writes
users: id, name, email
orders: id, user_id, total, created_at
order_items: id, order_id, product_id, quantity, price

# Query requires JOINs
SELECT u.name, o.total, COUNT(oi.id)
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
GROUP BY o.id;

# Denormalized - Good for reads
orders: id, user_id, user_name, user_email,
        total, item_count, created_at

# No JOINs needed
SELECT user_name, total, item_count FROM orders;

# Trade-off: user_name changes require updating all orders`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },

  // ===== SYSTEM DESIGN PATTERNS =====
  {
    id: 'sd-microservices-architecture',
    title: 'Microservices Architecture',
    category: 'Architecture Patterns',
    description: 'Decompose applications into small, independent services that communicate over APIs. Each service owns its data and can be deployed independently.',
    keyPoints: [
      'Single Responsibility: Each service does one thing well',
      'Independent deployment and scaling per service',
      'Technology diversity: Different languages/databases per service',
      'Challenges: Distributed transactions, network latency, debugging',
      'Requires: Service discovery, API gateway, monitoring',
    ],
    codeExample: `# Monolith vs Microservices

# Monolith
[Single Application]
- User Module
- Order Module
- Payment Module
- Inventory Module

# Microservices
[API Gateway] --> [User Service] --> [User DB]
      |
      +-------> [Order Service] --> [Order DB]
      |
      +-------> [Payment Service] --> [Payment DB]
      |
      +-------> [Inventory Service] --> [Inventory DB]

# Inter-service Communication
HTTP/REST: Simple, synchronous
gRPC: High performance, strongly typed
Message Queue: Async, decoupled (RabbitMQ, Kafka)`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-api-gateway',
    title: 'API Gateway Pattern',
    category: 'Architecture Patterns',
    description: 'Single entry point for all client requests. Handles cross-cutting concerns like authentication, rate limiting, and request routing.',
    keyPoints: [
      'Routes requests to appropriate microservices',
      'Handles auth, rate limiting, SSL termination',
      'Request/response transformation and aggregation',
      'Circuit breaker and retry logic',
      'Popular: Kong, AWS API Gateway, NGINX, Zuul',
    ],
    codeExample: `# API Gateway Responsibilities
Client Request --> [API Gateway]
                       |
          +------------+------------+
          |            |            |
   [Auth/JWT]    [Rate Limit]  [Logging]
          |
          +---> Route to Service

# Kong Configuration Example
services:
  - name: user-service
    url: http://user-svc:8080
    routes:
      - paths: ["/api/users"]
    plugins:
      - name: rate-limiting
        config:
          minute: 100
      - name: jwt`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-event-driven-architecture',
    title: 'Event-Driven Architecture',
    category: 'Architecture Patterns',
    description: 'Services communicate through events rather than direct calls. Enables loose coupling and high scalability.',
    keyPoints: [
      'Producers publish events, consumers subscribe',
      'Async communication enables better fault tolerance',
      'Event sourcing: Store events as source of truth',
      'CQRS: Separate read and write models',
      'Message brokers: Kafka, RabbitMQ, AWS SNS/SQS',
    ],
    codeExample: `# Event-Driven Order System
# 1. Order Service publishes event
event = {
    "type": "ORDER_PLACED",
    "data": {"order_id": 123, "user_id": 456, "total": 99.99}
}
kafka.publish("orders", event)

# 2. Multiple services consume independently
[Payment Service] --> Listens --> Process payment
[Inventory Service] --> Listens --> Reserve stock
[Notification Service] --> Listens --> Send confirmation
[Analytics Service] --> Listens --> Track metrics

# Event Sourcing
events = [
    {"type": "ITEM_ADDED", "product_id": 1, "qty": 2},
    {"type": "ITEM_REMOVED", "product_id": 1, "qty": 1},
    {"type": "CHECKOUT_COMPLETED"}
]
# Replay events to rebuild cart state`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-cqrs',
    title: 'CQRS Pattern',
    category: 'Architecture Patterns',
    description: 'Command Query Responsibility Segregation separates read and write operations into different models for optimized performance.',
    keyPoints: [
      'Commands: Write operations that change state',
      'Queries: Read operations that return data',
      'Different models optimized for each operation type',
      'Often combined with Event Sourcing',
      'Read model can be denormalized for fast queries',
    ],
    codeExample: `# CQRS Architecture
                    [API]
                   /     \\
            [Commands]  [Queries]
                |           |
          [Write Model] [Read Model]
                |           |
           [Primary DB] [Read Replica/Cache]
           (Normalized)  (Denormalized)

# Write Side (Command)
class CreateOrderCommand:
    user_id: int
    items: List[OrderItem]

def handle_create_order(cmd):
    order = Order.create(cmd.user_id, cmd.items)
    db.save(order)
    events.publish(OrderCreatedEvent(order))

# Read Side (Query)
class OrderSummaryQuery:
    user_id: int

def handle_order_summary(query):
    # Uses denormalized read-optimized table
    return cache.get(f"user:{query.user_id}:orders")`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-circuit-breaker',
    title: 'Circuit Breaker Pattern',
    category: 'Architecture Patterns',
    description: 'Prevent cascading failures by failing fast when a service is unhealthy. Three states: Closed, Open, Half-Open.',
    keyPoints: [
      'Closed: Normal operation, requests pass through',
      'Open: Failures exceeded threshold, requests fail immediately',
      'Half-Open: After timeout, allow limited requests to test',
      'Prevents overwhelming a struggling service',
      'Libraries: Hystrix, Resilience4j, Polly',
    ],
    codeExample: `# Circuit Breaker States
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.state = "CLOSED"
        self.timeout = timeout
        self.last_failure_time = None

    def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenException()

        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e

    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-saga-pattern',
    title: 'Saga Pattern',
    category: 'Architecture Patterns',
    description: 'Manage distributed transactions across microservices using a sequence of local transactions with compensating actions for rollback.',
    keyPoints: [
      'Choreography: Services react to events (decentralized)',
      'Orchestration: Central coordinator manages saga (centralized)',
      'Each step has a compensating transaction for rollback',
      'Eventually consistent (not ACID)',
      'Used when distributed transactions span multiple services',
    ],
    codeExample: `# Order Saga (Orchestration)
class OrderSaga:
    def execute(self, order):
        try:
            # Step 1: Reserve inventory
            inventory_service.reserve(order.items)

            # Step 2: Process payment
            payment_service.charge(order.user_id, order.total)

            # Step 3: Confirm order
            order_service.confirm(order.id)

        except PaymentFailedException:
            # Compensate: Release inventory
            inventory_service.release(order.items)
            raise

        except Exception:
            # Compensate all
            payment_service.refund(order.user_id, order.total)
            inventory_service.release(order.items)
            raise

# Choreography: Each service publishes/listens to events
[Order Created] --> [Inventory Reserved] --> [Payment Processed] --> [Order Confirmed]
       ^                    |                        |
       |          [Reserve Failed]          [Payment Failed]
       +------ Compensating Events -------+`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },

  // ===== MESSAGING & COMMUNICATION =====
  {
    id: 'sd-message-queues',
    title: 'Message Queues',
    category: 'Messaging',
    description: 'Asynchronous communication between services using queues. Decouples producers and consumers for reliability and scalability.',
    keyPoints: [
      'Point-to-point: One producer, one consumer per message',
      'Pub/Sub: One producer, multiple consumers',
      'Guaranteed delivery with acknowledgments',
      'Dead letter queues for failed messages',
      'Popular: RabbitMQ, AWS SQS, Redis Streams',
    ],
    codeExample: `# RabbitMQ Example
import pika

# Producer
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='orders')
channel.basic_publish(
    exchange='',
    routing_key='orders',
    body='{"order_id": 123}'
)

# Consumer
def callback(ch, method, properties, body):
    process_order(json.loads(body))
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue='orders', on_message_callback=callback)
channel.start_consuming()

# AWS SQS Example
sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(order))
messages = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10)`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-kafka',
    title: 'Apache Kafka',
    category: 'Messaging',
    description: 'Distributed streaming platform for high-throughput, fault-tolerant messaging. Ideal for event sourcing and real-time data pipelines.',
    keyPoints: [
      'Topics partitioned for parallel processing',
      'Messages persisted to disk (configurable retention)',
      'Consumer groups for load balancing',
      'Exactly-once semantics available',
      'High throughput: millions of messages/second',
    ],
    codeExample: `# Kafka Architecture
Producers --> [Topic: orders]
              [Partition 0] [Partition 1] [Partition 2]
                    |             |             |
              Consumer Group A (3 consumers)
              Consumer Group B (2 consumers)

# Python Producer
from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers='localhost:9092')
producer.send('orders', key=b'user123', value=b'{"order": 1}')

# Python Consumer
from kafka import KafkaConsumer
consumer = KafkaConsumer(
    'orders',
    group_id='order-processor',
    bootstrap_servers='localhost:9092'
)
for message in consumer:
    process(message.value)

# Key Concepts:
# - Offset: Position in partition
# - Replication Factor: Copies across brokers
# - Consumer Lag: How far behind consumer is`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-websockets',
    title: 'WebSockets & Real-time Communication',
    category: 'Messaging',
    description: 'Full-duplex communication channel over a single TCP connection. Essential for real-time features like chat, notifications, live updates.',
    keyPoints: [
      'Persistent connection (unlike HTTP request-response)',
      'Bi-directional: Both client and server can push messages',
      'Lower overhead than polling',
      'Scaling requires sticky sessions or pub/sub backend',
      'Alternatives: Server-Sent Events (SSE), Long Polling',
    ],
    codeExample: `# WebSocket Server (Python)
import websockets
import asyncio

connected = set()

async def handler(websocket, path):
    connected.add(websocket)
    try:
        async for message in websocket:
            # Broadcast to all connected clients
            for ws in connected:
                await ws.send(f"Broadcast: {message}")
    finally:
        connected.remove(websocket)

# Scaling WebSockets with Redis Pub/Sub
async def handler_with_redis(websocket, path):
    pubsub = redis.pubsub()
    await pubsub.subscribe("messages")

    # Listen for messages from Redis
    async for message in pubsub.listen():
        await websocket.send(message['data'])

# Client (JavaScript)
const ws = new WebSocket('wss://example.com/socket');
ws.onmessage = (event) => console.log(event.data);
ws.send('Hello Server!');`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-grpc',
    title: 'gRPC & Protocol Buffers',
    category: 'Messaging',
    description: 'High-performance RPC framework using Protocol Buffers for serialization. Excellent for microservice communication.',
    keyPoints: [
      'Binary serialization (smaller, faster than JSON)',
      'Strongly typed with code generation',
      'Supports streaming (unary, server, client, bidirectional)',
      'Built-in load balancing and deadline propagation',
      'HTTP/2 based with multiplexing',
    ],
    codeExample: `# Proto Definition (user.proto)
syntax = "proto3";

service UserService {
  rpc GetUser (GetUserRequest) returns (User);
  rpc ListUsers (ListUsersRequest) returns (stream User);
}

message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
}

# Python Server
class UserServicer(user_pb2_grpc.UserServiceServicer):
    def GetUser(self, request, context):
        user = db.get_user(request.id)
        return user_pb2.User(id=user.id, name=user.name)

# Python Client
channel = grpc.insecure_channel('localhost:50051')
stub = user_pb2_grpc.UserServiceStub(channel)
user = stub.GetUser(user_pb2.GetUserRequest(id=123))`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },

  // ===== STORAGE & DATA =====
  {
    id: 'sd-object-storage',
    title: 'Object Storage (S3, Blob)',
    category: 'Storage',
    description: 'Store unstructured data like images, videos, backups at massive scale with high durability.',
    keyPoints: [
      'Flat namespace with unique keys (not hierarchical)',
      '99.999999999% (11 9s) durability',
      'Cost-effective for large files',
      'Lifecycle policies for tiered storage (hot/cold)',
      'CDN integration for content delivery',
    ],
    codeExample: `# AWS S3 Operations (Python/boto3)
import boto3

s3 = boto3.client('s3')

# Upload file
s3.upload_file('local.jpg', 'my-bucket', 'images/photo.jpg')

# Generate presigned URL (temporary access)
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-bucket', 'Key': 'images/photo.jpg'},
    ExpiresIn=3600  # 1 hour
)

# Lifecycle Policy (auto-transition to cheaper storage)
{
    "Rules": [{
        "Status": "Enabled",
        "Transitions": [
            {"Days": 30, "StorageClass": "STANDARD_IA"},
            {"Days": 90, "StorageClass": "GLACIER"}
        ]
    }]
}`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-redis',
    title: 'Redis & In-Memory Caching',
    category: 'Storage',
    description: 'In-memory data store for caching, sessions, real-time analytics. Sub-millisecond latency.',
    keyPoints: [
      'Data structures: Strings, Lists, Sets, Hashes, Sorted Sets',
      'Use cases: Cache, Session store, Leaderboards, Rate limiting',
      'Persistence: RDB snapshots, AOF logging',
      'Clustering for horizontal scaling',
      'Pub/Sub for real-time messaging',
    ],
    codeExample: `import redis
r = redis.Redis(host='localhost', port=6379)

# Cache with TTL
r.setex('user:123', 3600, json.dumps(user_data))  # 1 hour TTL

# Rate Limiting (sliding window)
def is_rate_limited(user_id, limit=100, window=60):
    key = f"rate:{user_id}"
    current = r.incr(key)
    if current == 1:
        r.expire(key, window)
    return current > limit

# Leaderboard (Sorted Set)
r.zadd('leaderboard', {'player1': 100, 'player2': 85})
top_10 = r.zrevrange('leaderboard', 0, 9, withscores=True)

# Distributed Lock
lock = r.lock('resource_lock', timeout=10)
if lock.acquire(blocking=False):
    try:
        process_resource()
    finally:
        lock.release()`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-elasticsearch',
    title: 'Elasticsearch & Search',
    category: 'Storage',
    description: 'Distributed search and analytics engine. Excellent for full-text search, log analysis, and real-time analytics.',
    keyPoints: [
      'Inverted index for fast full-text search',
      'Near real-time indexing and search',
      'Horizontal scaling with sharding',
      'Aggregations for analytics',
      'Often paired with Logstash and Kibana (ELK stack)',
    ],
    codeExample: `# Index a document
PUT /products/_doc/1
{
  "name": "iPhone 15",
  "description": "Latest Apple smartphone with A17 chip",
  "price": 999,
  "category": "electronics"
}

# Full-text search with relevance scoring
GET /products/_search
{
  "query": {
    "multi_match": {
      "query": "smartphone apple",
      "fields": ["name^2", "description"]  // name has 2x weight
    }
  }
}

# Aggregation (faceted search)
GET /products/_search
{
  "aggs": {
    "categories": {
      "terms": { "field": "category.keyword" }
    },
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 100 },
          { "from": 100, "to": 500 },
          { "from": 500 }
        ]
      }
    }
  }
}`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-time-series-db',
    title: 'Time Series Databases',
    category: 'Storage',
    description: 'Optimized for time-stamped data like metrics, IoT sensor data, and financial data. High write throughput with efficient compression.',
    keyPoints: [
      'Optimized for append-only writes',
      'Automatic downsampling and retention policies',
      'Efficient compression for sequential data',
      'Built-in time-based aggregations',
      'Popular: InfluxDB, TimescaleDB, Prometheus',
    ],
    codeExample: `# InfluxDB Write (Line Protocol)
cpu,host=server01,region=us-west usage=45.2 1609459200000000000
cpu,host=server01,region=us-west usage=47.8 1609459260000000000

# InfluxDB Query (Flux)
from(bucket: "metrics")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu")
  |> filter(fn: (r) => r.host == "server01")
  |> aggregateWindow(every: 5m, fn: mean)

# Prometheus Query (PromQL)
# Average CPU usage over 5 minutes
avg_over_time(cpu_usage{host="server01"}[5m])

# 95th percentile response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Use cases:
# - Application metrics (latency, throughput)
# - Infrastructure monitoring (CPU, memory, disk)
# - IoT sensor data
# - Financial tick data`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },

  // ===== SECURITY =====
  {
    id: 'sd-authentication',
    title: 'Authentication & OAuth 2.0',
    category: 'Security',
    description: 'Verify user identity using various authentication methods. OAuth 2.0 is the industry standard for authorization.',
    keyPoints: [
      'Session-based: Server stores session, client stores cookie',
      'Token-based: Stateless JWT tokens',
      'OAuth 2.0 flows: Authorization Code, Client Credentials, PKCE',
      'OpenID Connect: Identity layer on top of OAuth 2.0',
      'Multi-factor authentication (MFA) for extra security',
    ],
    codeExample: `# JWT Token Structure
header.payload.signature

# Header
{"alg": "HS256", "typ": "JWT"}

# Payload
{
  "sub": "user123",
  "email": "user@example.com",
  "exp": 1609459200,
  "iat": 1609455600
}

# OAuth 2.0 Authorization Code Flow
1. User clicks "Login with Google"
2. Redirect to: https://accounts.google.com/oauth/authorize
   ?client_id=XXX&redirect_uri=XXX&scope=email&response_type=code
3. User authenticates with Google
4. Google redirects to: https://yourapp.com/callback?code=AUTH_CODE
5. Your server exchanges code for tokens:
   POST https://oauth2.googleapis.com/token
   {code, client_id, client_secret, redirect_uri}
6. Google returns: {access_token, refresh_token, id_token}`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-rate-limiting',
    title: 'Rate Limiting',
    category: 'Security',
    description: 'Protect APIs from abuse by limiting request frequency. Essential for preventing DDoS attacks and ensuring fair usage.',
    keyPoints: [
      'Token Bucket: Allows bursts up to bucket size',
      'Sliding Window: Smooth rate limiting',
      'Fixed Window: Simple but allows edge bursts',
      'Distributed rate limiting with Redis',
      'Return 429 Too Many Requests when exceeded',
    ],
    codeExample: `# Token Bucket Algorithm
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_update = time.time()

    def consume(self, tokens=1):
        now = time.time()
        # Refill tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.capacity,
                         self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# Redis Sliding Window
def is_allowed(user_id, limit=100, window=60):
    key = f"rate:{user_id}"
    now = time.time()

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, now - window)  # Remove old
    pipe.zadd(key, {str(now): now})              # Add current
    pipe.zcard(key)                               # Count
    pipe.expire(key, window)
    results = pipe.execute()

    return results[2] <= limit`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-encryption',
    title: 'Encryption (At Rest & In Transit)',
    category: 'Security',
    description: 'Protect data using encryption both when stored and during transmission.',
    keyPoints: [
      'In Transit: TLS/HTTPS encrypts network communication',
      'At Rest: Encrypt stored data (AES-256 common)',
      'Symmetric encryption: Same key for encrypt/decrypt (fast)',
      'Asymmetric encryption: Public/private keys (key exchange)',
      'Key management: Use KMS (AWS KMS, HashiCorp Vault)',
    ],
    codeExample: `# TLS/HTTPS Flow
1. Client Hello: Supported cipher suites, TLS version
2. Server Hello: Chosen cipher, server certificate
3. Client verifies certificate against trusted CAs
4. Key Exchange: Derive shared session key
5. Encrypted communication begins

# Encryption at Rest (Python)
from cryptography.fernet import Fernet

# Generate key (store securely!)
key = Fernet.generate_key()
f = Fernet(key)

# Encrypt
encrypted = f.encrypt(b"sensitive data")

# Decrypt
decrypted = f.decrypt(encrypted)

# AWS S3 Server-Side Encryption
s3.put_object(
    Bucket='my-bucket',
    Key='secret.txt',
    Body=data,
    ServerSideEncryption='aws:kms',
    SSEKMSKeyId='alias/my-key'
)`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-api-security',
    title: 'API Security Best Practices',
    category: 'Security',
    description: 'Secure your APIs against common attacks and vulnerabilities.',
    keyPoints: [
      'Input validation and sanitization',
      'Use HTTPS everywhere',
      'Implement proper authentication and authorization',
      'Rate limiting and throttling',
      'Security headers: CORS, CSP, HSTS',
    ],
    codeExample: `# Security Headers
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY

# Input Validation (Python/Pydantic)
from pydantic import BaseModel, EmailStr, validator

class CreateUserRequest(BaseModel):
    email: EmailStr
    name: str
    age: int

    @validator('name')
    def name_alphanumeric(cls, v):
        if not v.replace(' ', '').isalnum():
            raise ValueError('Name must be alphanumeric')
        return v

    @validator('age')
    def age_valid(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Invalid age')
        return v

# SQL Injection Prevention
# BAD
query = f"SELECT * FROM users WHERE id = {user_input}"

# GOOD (Parameterized)
cursor.execute("SELECT * FROM users WHERE id = ?", (user_input,))`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },

  // ===== RELIABILITY & AVAILABILITY =====
  {
    id: 'sd-high-availability',
    title: 'High Availability Design',
    category: 'Reliability',
    description: 'Design systems to minimize downtime. Measured in "nines" of availability (99.9% = 8.76 hours downtime/year).',
    keyPoints: [
      'Eliminate single points of failure (SPOF)',
      'Redundancy at every layer (servers, databases, network)',
      'Failover mechanisms: Active-passive, active-active',
      'Health checks and automatic recovery',
      'Geographic distribution for disaster recovery',
    ],
    codeExample: `# Availability Levels
99.9%   (3 nines) = 8.76 hours downtime/year
99.99%  (4 nines) = 52.6 minutes downtime/year
99.999% (5 nines) = 5.26 minutes downtime/year

# High Availability Architecture
                 [DNS - Route 53]
                 /              \\
          [Region A]         [Region B]
              |                   |
         [Load Balancer]    [Load Balancer]
         /    |    \\        /    |    \\
       [S1] [S2] [S3]     [S1] [S2] [S3]
              |                   |
    [Primary DB] <--replication--> [Standby DB]

# Health Check Example
def health_check():
    checks = {
        'database': check_db_connection(),
        'redis': check_redis_connection(),
        'disk': check_disk_space() > 10,  # 10% free
        'memory': check_memory_usage() < 90  # <90% used
    }
    healthy = all(checks.values())
    return {'status': 'healthy' if healthy else 'unhealthy', 'checks': checks}`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-disaster-recovery',
    title: 'Disaster Recovery & Backup',
    category: 'Reliability',
    description: 'Plan for catastrophic failures. Define RTO (Recovery Time Objective) and RPO (Recovery Point Objective).',
    keyPoints: [
      'RTO: Maximum acceptable downtime',
      'RPO: Maximum acceptable data loss',
      'Backup strategies: Full, incremental, differential',
      'Multi-region replication for geo-redundancy',
      'Regular disaster recovery drills',
    ],
    codeExample: `# RTO and RPO Examples
E-commerce Site:
  RTO: 1 hour (can afford 1 hour downtime)
  RPO: 5 minutes (can lose 5 minutes of orders)

Banking System:
  RTO: 15 minutes
  RPO: 0 (zero data loss - synchronous replication)

# Backup Strategy
Daily Full Backup (Sunday)
        |
Incremental (Mon, Tue, Wed, Thu, Fri, Sat)

# Recovery Steps
1. Restore latest full backup
2. Apply incremental backups in order
3. Replay transaction logs to RPO point

# Multi-Region DR Setup
Primary (us-east-1)          DR (us-west-2)
[App Servers]                [App Servers - Standby]
     |                              |
[RDS Primary] --async repl--> [RDS Read Replica]
     |                              |
[S3 Bucket] ---cross-region--> [S3 Replica]

# Failover: Promote DR to primary, update DNS`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-monitoring-observability',
    title: 'Monitoring & Observability',
    category: 'Reliability',
    description: 'The three pillars: Metrics, Logs, and Traces. Essential for understanding system behavior and debugging issues.',
    keyPoints: [
      'Metrics: Quantitative measurements (latency, throughput)',
      'Logs: Event records with context',
      'Traces: Request flow across services',
      'Alerting based on thresholds and anomalies',
      'Dashboards for visualization',
    ],
    codeExample: `# Metrics (Prometheus/Grafana)
# Counter - always increasing
http_requests_total{method="GET", status="200"} 1234

# Gauge - can go up/down
temperature_celsius{location="server-room"} 23.5

# Histogram - distribution
http_request_duration_seconds_bucket{le="0.1"} 500
http_request_duration_seconds_bucket{le="0.5"} 800

# Structured Logging (JSON)
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "ERROR",
  "service": "order-service",
  "trace_id": "abc123",
  "user_id": "user456",
  "message": "Payment failed",
  "error": "Card declined"
}

# Distributed Tracing (OpenTelemetry)
Trace ID: abc123
  [API Gateway] ----50ms----->
    [Order Service] ----20ms----->
      [Inventory Service] ----10ms----->
      [Payment Service] ----100ms-----> (SLOW!)
        [Stripe API] ----80ms----->`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-graceful-degradation',
    title: 'Graceful Degradation',
    category: 'Reliability',
    description: 'When parts of the system fail, continue providing core functionality with reduced capabilities instead of complete failure.',
    keyPoints: [
      'Prioritize critical features over nice-to-haves',
      'Serve cached/stale content when fresh data unavailable',
      'Feature flags to disable non-essential features',
      'Fallback responses when services are down',
      'Inform users about degraded state',
    ],
    codeExample: `# Graceful Degradation Examples

# 1. Cache Fallback
def get_product(product_id):
    try:
        return product_service.get(product_id)
    except ServiceUnavailable:
        # Return cached version (may be stale)
        cached = cache.get(f"product:{product_id}")
        if cached:
            return {**cached, "_stale": True}
        raise

# 2. Feature Flag
def get_recommendations(user_id):
    if not feature_flags.is_enabled('recommendations'):
        return []  # Empty recommendations instead of error
    return recommendation_service.get(user_id)

# 3. Timeout with Default
async def get_personalized_feed(user_id):
    try:
        return await asyncio.wait_for(
            personalization_service.get_feed(user_id),
            timeout=2.0  # 2 second timeout
        )
    except asyncio.TimeoutError:
        # Fall back to generic popular content
        return get_popular_content()

# 4. Partial Response
def get_product_page(product_id):
    product = get_product(product_id)  # Critical
    try:
        reviews = get_reviews(product_id)  # Non-critical
    except:
        reviews = []
    return {"product": product, "reviews": reviews}`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },

  // ===== COMMON SYSTEM DESIGNS =====
  {
    id: 'sd-url-shortener',
    title: 'Design: URL Shortener',
    category: 'System Design Examples',
    description: 'Design a service like bit.ly that shortens long URLs and redirects users.',
    keyPoints: [
      'Base62 encoding for short codes (a-z, A-Z, 0-9)',
      'Use counter or hash for unique ID generation',
      'Consider custom aliases and collision handling',
      'Analytics: Click tracking, geographic data',
      'Cache hot URLs, set TTL for expiration',
    ],
    codeExample: `# URL Shortener Design
Components:
1. API Service (create/redirect)
2. ID Generator (unique short codes)
3. Database (URL mappings)
4. Cache (hot URLs)
5. Analytics Service

# Short Code Generation
BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def encode_base62(num):
    result = []
    while num:
        result.append(BASE62[num % 62])
        num //= 62
    return ''.join(reversed(result)) or '0'

# Counter-based approach
counter = 100000000  # Start from 100M
short_code = encode_base62(counter)  # "6LAze"

# API Flow
POST /shorten {"long_url": "https://example.com/very/long/url"}
-> Generate short_code
-> Store in DB: {short_code, long_url, created_at, user_id}
-> Return: "https://short.ly/6LAze"

GET /6LAze
-> Check cache -> Check DB -> Redirect to long_url
-> Async: Log click analytics`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-rate-limiter-design',
    title: 'Design: Rate Limiter',
    category: 'System Design Examples',
    description: 'Design a distributed rate limiting system to protect APIs from abuse.',
    keyPoints: [
      'Algorithms: Token Bucket, Sliding Window, Fixed Window',
      'Distributed implementation with Redis',
      'Handle race conditions with atomic operations',
      'Different limits per user tier/endpoint',
      'Return Retry-After header when limited',
    ],
    codeExample: `# Distributed Rate Limiter Design
Components:
1. Rate Limiter Service
2. Redis Cluster (centralized counters)
3. Rules Engine (configurable limits)

# Sliding Window Log (Redis)
def is_allowed(user_id, limit=100, window=60):
    key = f"rate:{user_id}"
    now = time.time()

    # Atomic Redis transaction
    lua_script = """
    redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1])
    local count = redis.call('ZCARD', KEYS[1])
    if count < tonumber(ARGV[2]) then
        redis.call('ZADD', KEYS[1], ARGV[3], ARGV[3])
        redis.call('EXPIRE', KEYS[1], ARGV[4])
        return 1
    end
    return 0
    """
    return redis.eval(lua_script, 1, key,
                     now - window, limit, now, window)

# Tiered Limits
LIMITS = {
    "free": {"requests_per_minute": 10},
    "basic": {"requests_per_minute": 100},
    "premium": {"requests_per_minute": 1000}
}

# Response when limited
HTTP 429 Too Many Requests
{
  "error": "Rate limit exceeded",
  "retry_after": 30
}
Headers: Retry-After: 30`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-notification-system',
    title: 'Design: Notification System',
    category: 'System Design Examples',
    description: 'Design a system to send notifications via push, SMS, email across millions of users.',
    keyPoints: [
      'Multiple channels: Push, SMS, Email, In-app',
      'User preferences and opt-out handling',
      'Prioritization and rate limiting per user',
      'Templating and personalization',
      'Delivery tracking and retry logic',
    ],
    codeExample: `# Notification System Architecture
Components:
1. Notification Service (API)
2. Message Queue (Kafka)
3. Channel Workers (Push, SMS, Email)
4. Template Service
5. Preference Service
6. Delivery Tracker

# Flow
Event Producer --> [Kafka: notifications]
                        |
              [Priority Router]
              /      |       \\
      [Push]    [Email]    [SMS]
      Worker    Worker    Worker
         |         |         |
      [APNs]   [SendGrid]  [Twilio]

# Notification Request
{
  "user_id": "123",
  "template": "order_shipped",
  "data": {"order_id": "456", "tracking": "ABC123"},
  "channels": ["push", "email"],
  "priority": "high"
}

# User Preferences
{
  "user_id": "123",
  "push_enabled": true,
  "email_enabled": true,
  "sms_enabled": false,
  "quiet_hours": {"start": "22:00", "end": "08:00"}
}

# Retry with exponential backoff
attempts = [0, 1min, 5min, 30min, 2hr, 12hr]`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-chat-system',
    title: 'Design: Real-time Chat System',
    category: 'System Design Examples',
    description: 'Design a messaging platform like WhatsApp or Slack supporting 1:1 and group chats.',
    keyPoints: [
      'WebSocket connections for real-time messaging',
      'Message queue for guaranteed delivery',
      'Presence system (online/offline status)',
      'Message persistence and sync across devices',
      'Read receipts and typing indicators',
    ],
    codeExample: `# Chat System Architecture
Components:
1. Chat Service (WebSocket servers)
2. Message Store (Cassandra)
3. Presence Service (Redis)
4. Push Notification Service
5. Media Storage (S3)

# WebSocket Connection Management
User connects --> [Load Balancer]
                      |
              [Chat Server Pool]
                      |
[Redis] <-- Store: user_id -> server_id mapping

# Message Flow (1:1)
Alice sends message to Bob:
1. Alice's WebSocket --> Chat Server A
2. Publish to Kafka topic "messages"
3. Store in Cassandra (sender, receiver, content, timestamp)
4. Lookup Bob's connection server in Redis
5. If online: Forward to Chat Server B --> Bob's WebSocket
6. If offline: Queue for push notification

# Message Schema (Cassandra)
CREATE TABLE messages (
    chat_id UUID,
    message_id TIMEUUID,
    sender_id UUID,
    content TEXT,
    created_at TIMESTAMP,
    PRIMARY KEY (chat_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);

# Presence (Redis)
HSET presence user123 "{server: 'ws1', last_seen: 1234567890}"
EXPIRE presence:user123 60  # TTL for online status`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-newsfeed',
    title: 'Design: Social Media Newsfeed',
    category: 'System Design Examples',
    description: 'Design a feed system like Facebook or Twitter that shows personalized content.',
    keyPoints: [
      'Fan-out on write vs Fan-out on read',
      'Hybrid approach: Push for normal users, pull for celebrities',
      'Ranking and personalization algorithms',
      'Caching feed with cache invalidation strategy',
      'Pagination with cursor-based approach',
    ],
    codeExample: `# Newsfeed Architecture
Components:
1. Post Service (create posts)
2. Fanout Service (distribute to followers)
3. Feed Service (retrieve user feed)
4. Ranking Service (personalization)
5. Cache (Redis)

# Fan-out Strategies

# Fan-out on Write (Push Model)
# Good for users with few followers
User posts --> Get all followers --> Write to each follower's feed cache

# Fan-out on Read (Pull Model)
# Good for celebrities with millions of followers
User requests feed --> Fetch posts from followed users --> Merge and rank

# Hybrid Approach
if user.follower_count < 10000:
    fanout_on_write(post)
else:
    mark_for_pull(post)

# Feed Retrieval
def get_feed(user_id, cursor=None, limit=20):
    # Get from cache
    cached_feed = redis.zrevrange(f"feed:{user_id}", 0, limit)

    # Merge with celebrity posts (pull)
    celebrity_posts = get_celebrity_posts(user_id.following)

    # Rank and personalize
    feed = rank(cached_feed + celebrity_posts)

    return feed[:limit]

# Cursor-based pagination
{"posts": [...], "next_cursor": "post_id_123"}`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-uber-design',
    title: 'Design: Ride-Sharing Service',
    category: 'System Design Examples',
    description: 'Design a system like Uber or Lyft for matching riders with drivers in real-time.',
    keyPoints: [
      'Location tracking and geospatial indexing',
      'Real-time matching algorithm',
      'ETA calculation and route optimization',
      'Surge pricing based on demand',
      'Driver and rider apps with different features',
    ],
    codeExample: `# Ride-Sharing Architecture
Components:
1. Ride Service (booking, status)
2. Location Service (real-time tracking)
3. Matching Service (driver assignment)
4. Pricing Service (fare calculation)
5. Payment Service
6. Notification Service

# Geospatial Indexing (GeoHash/QuadTree)
# Divide world into grids for efficient nearby search
def find_nearby_drivers(lat, lng, radius_km):
    geohash = encode_geohash(lat, lng, precision=6)
    neighbors = get_neighbor_geohashes(geohash)

    drivers = []
    for gh in [geohash] + neighbors:
        drivers.extend(redis.smembers(f"drivers:{gh}"))

    return filter_by_distance(drivers, lat, lng, radius_km)

# Matching Algorithm
def match_rider_to_driver(rider_location, ride_type):
    nearby_drivers = find_nearby_drivers(
        rider_location.lat,
        rider_location.lng,
        radius_km=5
    )

    available = [d for d in nearby_drivers
                 if d.status == 'available'
                 and d.vehicle_type == ride_type]

    # Score by distance, rating, acceptance rate
    scored = [(driver, calculate_score(driver, rider_location))
              for driver in available]

    return sorted(scored, key=lambda x: x[1], reverse=True)[0]

# Location Updates (every 5 seconds)
Driver App --> WebSocket --> Location Service --> Redis GeoHash`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-youtube-design',
    title: 'Design: Video Streaming Platform',
    category: 'System Design Examples',
    description: 'Design a video platform like YouTube supporting upload, transcoding, and streaming.',
    keyPoints: [
      'Video transcoding to multiple resolutions/formats',
      'Adaptive bitrate streaming (HLS, DASH)',
      'CDN for global video delivery',
      'Thumbnail generation and video processing',
      'View counting and analytics at scale',
    ],
    codeExample: `# Video Platform Architecture
Components:
1. Upload Service
2. Transcoding Pipeline
3. Video Storage (S3)
4. Metadata Service (DB)
5. Streaming Service
6. CDN (CloudFront)
7. Analytics Service

# Upload Flow
1. Client requests pre-signed upload URL
2. Client uploads directly to S3
3. S3 event triggers transcoding pipeline
4. Transcode to: 360p, 480p, 720p, 1080p, 4K
5. Generate thumbnails at multiple timestamps
6. Update metadata: status = "ready"

# Adaptive Bitrate Streaming (HLS)
video.m3u8 (master playlist)
├── video_360p.m3u8 --> video_360p_001.ts, video_360p_002.ts...
├── video_720p.m3u8 --> video_720p_001.ts, video_720p_002.ts...
└── video_1080p.m3u8 --> video_1080p_001.ts, video_1080p_002.ts...

# Player automatically switches quality based on bandwidth

# View Counting (avoid hotspot)
# Don't increment DB on every view
def record_view(video_id):
    # Aggregate in Redis first
    redis.incr(f"views:{video_id}")

# Batch flush to DB every minute
def flush_view_counts():
    for key in redis.scan("views:*"):
        video_id = key.split(":")[1]
        count = redis.getset(key, 0)
        db.execute("UPDATE videos SET views = views + ? WHERE id = ?",
                   count, video_id)`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-search-engine',
    title: 'Design: Search Engine',
    category: 'System Design Examples',
    description: 'Design a search system like Google or Elasticsearch for web or product search.',
    keyPoints: [
      'Web crawler for content discovery',
      'Inverted index for fast text search',
      'PageRank or relevance scoring',
      'Query parsing and spell correction',
      'Caching frequent queries',
    ],
    codeExample: `# Search Engine Architecture
Components:
1. Web Crawler
2. Document Processor
3. Indexer (build inverted index)
4. Query Service
5. Ranking Service

# Inverted Index
Document 1: "The quick brown fox"
Document 2: "The lazy dog"
Document 3: "Quick brown dog"

Index:
"the"   -> [1, 2]
"quick" -> [1, 3]
"brown" -> [1, 3]
"fox"   -> [1]
"lazy"  -> [2]
"dog"   -> [2, 3]

# Search "quick brown"
quick -> [1, 3]
brown -> [1, 3]
Intersection: [1, 3] -> Rank by relevance

# TF-IDF Scoring
TF = term frequency in document
IDF = log(total docs / docs containing term)
Score = TF * IDF

# Query Processing
def search(query):
    # 1. Tokenize and normalize
    tokens = tokenize(query.lower())

    # 2. Spell correction
    tokens = [spell_correct(t) for t in tokens]

    # 3. Get posting lists
    postings = [index.get(t) for t in tokens]

    # 4. Intersect and rank
    candidates = intersect(postings)
    ranked = rank_by_relevance(candidates, tokens)

    return ranked[:10]`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-payment-system',
    title: 'Design: Payment System',
    category: 'System Design Examples',
    description: 'Design a payment processing system handling transactions securely and reliably.',
    keyPoints: [
      'Idempotency keys to prevent duplicate charges',
      'Two-phase commit or saga pattern for transactions',
      'PCI DSS compliance for card data',
      'Fraud detection and risk scoring',
      'Reconciliation and settlement processes',
    ],
    codeExample: `# Payment System Architecture
Components:
1. Payment Gateway API
2. Payment Processor
3. Fraud Detection Service
4. Ledger Service (double-entry)
5. Notification Service

# Idempotent Payment Request
POST /payments
Headers: Idempotency-Key: unique-uuid-12345
{
  "amount": 9999,
  "currency": "USD",
  "payment_method_id": "pm_123",
  "merchant_id": "merchant_456"
}

def process_payment(idempotency_key, request):
    # Check if already processed
    existing = db.get_by_idempotency_key(idempotency_key)
    if existing:
        return existing.result  # Return same result

    # Process payment
    result = payment_processor.charge(request)

    # Store with idempotency key
    db.save(idempotency_key, result)
    return result

# Double-Entry Ledger
# Debit Customer Account, Credit Merchant Account
transactions:
| id | account_id | type   | amount | balance_after |
| 1  | customer   | DEBIT  | -99.99 | 400.01        |
| 2  | merchant   | CREDIT | +99.99 | 1099.99       |

# Always: sum(debits) = sum(credits)

# Payment State Machine
PENDING -> AUTHORIZED -> CAPTURED -> SETTLED
    |           |
    v           v
  FAILED    REFUNDED`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-ecommerce-design',
    title: 'Design: E-commerce Platform',
    category: 'System Design Examples',
    description: 'Design a scalable e-commerce system like Amazon handling products, orders, and inventory.',
    keyPoints: [
      'Product catalog with search and filtering',
      'Shopping cart (guest and authenticated)',
      'Inventory management and reservation',
      'Order processing pipeline',
      'Recommendation engine',
    ],
    codeExample: `# E-commerce Architecture
Components:
1. Product Catalog Service
2. Search Service (Elasticsearch)
3. Cart Service
4. Inventory Service
5. Order Service
6. Payment Service
7. Shipping Service

# Order Flow
1. User adds items to cart
2. User initiates checkout
3. Reserve inventory (with TTL)
4. Process payment
5. Confirm order
6. Release reserved inventory to "sold"
7. Trigger fulfillment

# Inventory Reservation (prevent overselling)
def checkout(cart):
    order_id = generate_order_id()

    # Reserve inventory with TTL
    for item in cart.items:
        success = inventory_service.reserve(
            sku=item.sku,
            quantity=item.quantity,
            reservation_id=order_id,
            ttl_minutes=10
        )
        if not success:
            rollback_reservations(order_id)
            raise OutOfStockError(item.sku)

    # Process payment
    payment_result = payment_service.charge(cart.total)

    if payment_result.success:
        inventory_service.confirm_reservation(order_id)
        return create_order(cart, order_id)
    else:
        inventory_service.cancel_reservation(order_id)
        raise PaymentFailedError()

# Cart Service (Redis)
HSET cart:user123 sku:ABC quantity:2 added_at:123456
EXPIRE cart:user123 604800  # 7 days TTL`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },

  // ===== INFRASTRUCTURE =====
  {
    id: 'sd-containerization',
    title: 'Containerization & Docker',
    category: 'Infrastructure',
    description: 'Package applications with dependencies into portable containers for consistent deployment across environments.',
    keyPoints: [
      'Containers share OS kernel (lighter than VMs)',
      'Dockerfile defines image build steps',
      'Images are immutable, containers are instances',
      'Docker Compose for multi-container apps',
      'Container registries: Docker Hub, ECR, GCR',
    ],
    codeExample: `# Dockerfile Example
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]

# Docker Compose
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://db:5432/myapp
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-kubernetes',
    title: 'Kubernetes Orchestration',
    category: 'Infrastructure',
    description: 'Container orchestration platform for deploying, scaling, and managing containerized applications.',
    keyPoints: [
      'Pods: Smallest deployable unit (1+ containers)',
      'Deployments: Declarative updates, rolling deploys',
      'Services: Stable networking, load balancing',
      'ConfigMaps/Secrets: Configuration management',
      'Horizontal Pod Autoscaler (HPA) for auto-scaling',
    ],
    codeExample: `# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
      - name: api
        image: myapp:v1.2.3
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
---
# Service (Load Balancer)
apiVersion: v1
kind: Service
metadata:
  name: api-server
spec:
  type: LoadBalancer
  selector:
    app: api-server
  ports:
  - port: 80
    targetPort: 8000
---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-ci-cd',
    title: 'CI/CD Pipelines',
    category: 'Infrastructure',
    description: 'Continuous Integration and Continuous Deployment automate building, testing, and deploying code changes.',
    keyPoints: [
      'CI: Automatically build and test on every commit',
      'CD: Automatically deploy to staging/production',
      'Pipeline stages: Build, Test, Security Scan, Deploy',
      'Blue-green and canary deployment strategies',
      'Tools: GitHub Actions, GitLab CI, Jenkins, CircleCI',
    ],
    codeExample: `# GitHub Actions CI/CD Pipeline
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest --cov=app tests/

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install safety bandit
      - run: safety check
      - run: bandit -r app/

  deploy:
    needs: [test, security]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker image
        run: |
          docker build -t myapp:\${{ github.sha }} .
          docker push myregistry/myapp:\${{ github.sha }}
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/api-server \\
            api=myregistry/myapp:\${{ github.sha }}
          kubectl rollout status deployment/api-server`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-service-discovery',
    title: 'Service Discovery',
    category: 'Infrastructure',
    description: 'Automatically detect and connect to services in a dynamic microservices environment.',
    keyPoints: [
      'Client-side: Client queries registry and load balances',
      'Server-side: Load balancer queries registry',
      'Health checking for service availability',
      'DNS-based vs registry-based discovery',
      'Tools: Consul, Eureka, Kubernetes DNS',
    ],
    codeExample: `# Service Discovery Patterns

# 1. DNS-Based (Kubernetes)
# Service automatically gets DNS name
api-service.namespace.svc.cluster.local

# Client simply uses DNS
requests.get("http://api-service:8080/users")

# 2. Registry-Based (Consul)
# Service Registration
{
  "Name": "api-service",
  "Address": "10.0.1.5",
  "Port": 8080,
  "Check": {
    "HTTP": "http://10.0.1.5:8080/health",
    "Interval": "10s"
  }
}

# Service Discovery Query
GET /v1/health/service/api-service?passing=true

# 3. Client-Side Load Balancing
class ServiceClient:
    def __init__(self, service_name):
        self.service_name = service_name
        self.instances = []
        self.refresh_instances()

    def refresh_instances(self):
        self.instances = consul.get_healthy(self.service_name)

    def call(self, endpoint):
        instance = random.choice(self.instances)
        return requests.get(f"http://{instance.address}:{instance.port}{endpoint}")`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-dns',
    title: 'DNS & Domain Management',
    category: 'Infrastructure',
    description: 'Domain Name System translates domain names to IP addresses. Critical for routing traffic globally.',
    keyPoints: [
      'Record types: A, AAAA, CNAME, MX, TXT',
      'TTL affects propagation time and cache',
      'GeoDNS for location-based routing',
      'DNS load balancing and failover',
      'DNSSEC for security',
    ],
    codeExample: `# DNS Record Types
# A Record - IPv4 address
example.com.  300  A  192.168.1.1

# AAAA Record - IPv6 address
example.com.  300  AAAA  2001:db8::1

# CNAME - Alias to another domain
www.example.com.  300  CNAME  example.com.

# MX - Mail server
example.com.  300  MX  10 mail.example.com.

# TXT - Verification, SPF, etc
example.com.  300  TXT  "v=spf1 include:_spf.google.com ~all"

# DNS Resolution Flow
Browser --> Local DNS Cache
        --> ISP DNS Resolver
        --> Root DNS (.com)
        --> TLD DNS (example.com)
        --> Authoritative DNS (IP address)

# Route 53 Health Check & Failover
Primary: us-east-1 (active)
Failover: us-west-2 (passive)

If health check fails on primary:
DNS automatically routes to failover

# GeoDNS Example
User in Europe -> eu-west-1.example.com
User in Asia -> ap-southeast-1.example.com
User in US -> us-east-1.example.com`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },

  // ===== DATA PROCESSING =====
  {
    id: 'sd-batch-processing',
    title: 'Batch Processing',
    category: 'Data Processing',
    description: 'Process large volumes of data in scheduled jobs. Suitable for analytics, ETL, and periodic computations.',
    keyPoints: [
      'Process data in chunks, not real-time',
      'MapReduce paradigm for distributed processing',
      'Idempotent jobs for reliability',
      'Tools: Apache Spark, Hadoop, AWS EMR',
      'Schedule with Airflow, cron, or Step Functions',
    ],
    codeExample: `# Apache Spark Batch Job
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DailyAnalytics").getOrCreate()

# Read from data lake
df = spark.read.parquet("s3://data-lake/events/2024-01-15/")

# Transform
daily_stats = df \\
    .groupBy("user_id", "event_type") \\
    .agg(
        count("*").alias("event_count"),
        avg("duration").alias("avg_duration")
    )

# Write results
daily_stats.write \\
    .mode("overwrite") \\
    .parquet("s3://data-warehouse/daily_stats/2024-01-15/")

# Airflow DAG for scheduling
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG('daily_analytics', schedule_interval='@daily') as dag:
    extract = PythonOperator(task_id='extract', python_callable=extract_data)
    transform = PythonOperator(task_id='transform', python_callable=transform_data)
    load = PythonOperator(task_id='load', python_callable=load_data)

    extract >> transform >> load`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-stream-processing',
    title: 'Stream Processing',
    category: 'Data Processing',
    description: 'Process data in real-time as it arrives. Essential for live dashboards, fraud detection, and real-time recommendations.',
    keyPoints: [
      'Process events as they happen (low latency)',
      'Windowing: Tumbling, sliding, session windows',
      'State management for aggregations',
      'Exactly-once semantics',
      'Tools: Kafka Streams, Apache Flink, Spark Streaming',
    ],
    codeExample: `# Kafka Streams Example (Java-style pseudocode)
# Real-time click analytics

StreamsBuilder builder = new StreamsBuilder();

# Read from topic
KStream<String, ClickEvent> clicks =
    builder.stream("clicks");

# Window aggregation (5-minute tumbling window)
KTable<Windowed<String>, Long> clickCounts = clicks
    .groupBy((key, click) -> click.getPageId())
    .windowedBy(TimeWindows.ofSizeWithNoGrace(Duration.ofMinutes(5)))
    .count();

# Write results
clickCounts.toStream()
    .to("page-click-counts");

# Apache Flink Python Example
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# Define source (Kafka)
t_env.execute_sql("""
    CREATE TABLE clicks (
        user_id STRING,
        page_id STRING,
        event_time TIMESTAMP(3),
        WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'clicks',
        'format' = 'json'
    )
""")

# Real-time aggregation
result = t_env.sql_query("""
    SELECT
        page_id,
        TUMBLE_START(event_time, INTERVAL '5' MINUTE) as window_start,
        COUNT(*) as click_count
    FROM clicks
    GROUP BY page_id, TUMBLE(event_time, INTERVAL '5' MINUTE)
""")`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-data-warehouse',
    title: 'Data Warehouse & OLAP',
    category: 'Data Processing',
    description: 'Centralized repository optimized for analytical queries across large datasets.',
    keyPoints: [
      'OLAP vs OLTP: Analytics vs Transactions',
      'Star/Snowflake schema for dimensional modeling',
      'Columnar storage for fast aggregations',
      'ETL/ELT pipelines to load data',
      'Tools: Snowflake, BigQuery, Redshift, Databricks',
    ],
    codeExample: `# Star Schema Design
# Fact Table (measures/metrics)
fact_orders:
| order_id | date_id | product_id | customer_id | quantity | amount |

# Dimension Tables (descriptive attributes)
dim_date:
| date_id | date | month | quarter | year | is_holiday |

dim_product:
| product_id | name | category | brand | price |

dim_customer:
| customer_id | name | city | state | segment |

# Analytical Query (slice and dice)
SELECT
    d.year,
    d.quarter,
    p.category,
    SUM(f.amount) as revenue,
    COUNT(DISTINCT f.customer_id) as unique_customers
FROM fact_orders f
JOIN dim_date d ON f.date_id = d.date_id
JOIN dim_product p ON f.product_id = p.product_id
WHERE d.year = 2024
GROUP BY d.year, d.quarter, p.category
ORDER BY revenue DESC;

# ETL Pipeline
Source (OLTP DBs) --> Extract
                        |
                    Transform (clean, join, aggregate)
                        |
                      Load --> Data Warehouse
                        |
                    BI Tools (Tableau, Looker, Power BI)`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-data-lake',
    title: 'Data Lake Architecture',
    category: 'Data Processing',
    description: 'Store raw, unstructured data at scale. Schema-on-read approach allows flexibility for diverse data types.',
    keyPoints: [
      'Store raw data in native format',
      'Schema-on-read vs Schema-on-write',
      'Zones: Raw, Cleaned, Curated',
      'Metadata catalog for discoverability',
      'Tools: S3 + Glue, Delta Lake, Apache Iceberg',
    ],
    codeExample: `# Data Lake Architecture
S3 Data Lake Structure:
s3://data-lake/
├── raw/                    # Ingested as-is
│   ├── clickstream/
│   │   └── 2024/01/15/
│   ├── transactions/
│   └── logs/
├── cleaned/               # Validated, deduplicated
│   ├── clickstream/
│   └── transactions/
├── curated/               # Business-ready datasets
│   ├── customer_360/
│   └── product_analytics/
└── sandbox/               # Ad-hoc analysis

# Delta Lake (ACID on Data Lake)
from delta import DeltaTable

# Write with schema enforcement
df.write.format("delta") \\
    .mode("append") \\
    .partitionBy("date") \\
    .save("s3://data-lake/curated/orders/")

# Time travel (query historical versions)
df = spark.read.format("delta") \\
    .option("versionAsOf", 5) \\
    .load("s3://data-lake/curated/orders/")

# MERGE (upsert)
deltaTable = DeltaTable.forPath(spark, "s3://data-lake/curated/customers/")
deltaTable.alias("target").merge(
    updates.alias("source"),
    "target.customer_id = source.customer_id"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },

  // ===== API DESIGN =====
  {
    id: 'sd-rest-api-design',
    title: 'REST API Design',
    category: 'API Design',
    description: 'Design intuitive, consistent REST APIs following best practices for resource-oriented architecture.',
    keyPoints: [
      'Use nouns for resources, HTTP verbs for actions',
      'Consistent naming: plural nouns, kebab-case',
      'Proper status codes: 200, 201, 400, 401, 404, 500',
      'Pagination, filtering, sorting for collections',
      'HATEOAS for discoverability',
    ],
    codeExample: `# REST API Best Practices

# Resources (nouns, plural)
GET    /users              # List users
POST   /users              # Create user
GET    /users/123          # Get user
PUT    /users/123          # Update user (full)
PATCH  /users/123          # Update user (partial)
DELETE /users/123          # Delete user

# Nested resources
GET    /users/123/orders   # User's orders
POST   /users/123/orders   # Create order for user

# Filtering, Sorting, Pagination
GET /products?category=electronics&sort=-price&page=2&limit=20

# Response format
{
  "data": [...],
  "meta": {
    "total": 100,
    "page": 2,
    "limit": 20,
    "pages": 5
  },
  "links": {
    "self": "/products?page=2",
    "next": "/products?page=3",
    "prev": "/products?page=1"
  }
}

# Error Response
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": [
      {"field": "email", "message": "Invalid email format"}
    ]
  }
}`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-graphql',
    title: 'GraphQL API Design',
    category: 'API Design',
    description: 'Query language for APIs that lets clients request exactly the data they need, reducing over-fetching.',
    keyPoints: [
      'Single endpoint, client specifies data shape',
      'Strongly typed schema',
      'Queries for reads, Mutations for writes',
      'Subscriptions for real-time updates',
      'N+1 problem: Use DataLoader for batching',
    ],
    codeExample: `# GraphQL Schema
type User {
  id: ID!
  name: String!
  email: String!
  orders: [Order!]!
}

type Order {
  id: ID!
  total: Float!
  items: [OrderItem!]!
  createdAt: DateTime!
}

type Query {
  user(id: ID!): User
  users(limit: Int, offset: Int): [User!]!
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
}

# Client Query (get exactly what you need)
query GetUserWithOrders {
  user(id: "123") {
    name
    email
    orders {
      id
      total
      items {
        productName
        quantity
      }
    }
  }
}

# DataLoader to prevent N+1
from promise import Promise
from promise.dataloader import DataLoader

class UserLoader(DataLoader):
    def batch_load_fn(self, user_ids):
        users = User.objects.filter(id__in=user_ids)
        user_map = {u.id: u for u in users}
        return Promise.resolve([user_map.get(id) for id in user_ids])`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-api-versioning',
    title: 'API Versioning Strategies',
    category: 'API Design',
    description: 'Manage API changes without breaking existing clients. Choose a versioning strategy that fits your needs.',
    keyPoints: [
      'URL versioning: /v1/users, /v2/users',
      'Header versioning: Accept: application/vnd.api.v2+json',
      'Query param: /users?version=2',
      'Semantic versioning for breaking changes',
      'Deprecation policy and sunset headers',
    ],
    codeExample: `# API Versioning Approaches

# 1. URL Path Versioning (most common)
GET /v1/users/123
GET /v2/users/123

# 2. Header Versioning
GET /users/123
Headers:
  Accept: application/vnd.myapi.v2+json

# 3. Query Parameter
GET /users/123?api-version=2

# Deprecation Headers
HTTP/1.1 200 OK
Deprecation: true
Sunset: Sat, 31 Dec 2024 23:59:59 GMT
Link: </v2/users>; rel="successor-version"

# Version Negotiation (Flask example)
from flask import request

@app.route('/users/<id>')
def get_user(id):
    version = request.headers.get('Api-Version', '1')

    if version == '2':
        return get_user_v2(id)
    else:
        return get_user_v1(id)

# Best Practices:
# - Support N-1 versions (current + previous)
# - Communicate deprecation timeline clearly
# - Provide migration guides
# - Use feature flags for gradual rollout`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },

  // ===== DISTRIBUTED SYSTEMS CONCEPTS =====
  {
    id: 'sd-consistent-hashing',
    title: 'Consistent Hashing',
    category: 'Distributed Systems',
    description: 'Distribute data across nodes with minimal redistribution when nodes are added or removed.',
    keyPoints: [
      'Hash both keys and nodes onto a ring',
      'Key assigned to first node clockwise',
      'Adding/removing node only affects neighbors',
      'Virtual nodes for better distribution',
      'Used in: DynamoDB, Cassandra, Redis Cluster',
    ],
    codeExample: `# Consistent Hashing Implementation
import hashlib

class ConsistentHash:
    def __init__(self, nodes=None, virtual_nodes=100):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)
        self.sorted_keys.sort()

    def remove_node(self, node):
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}:{i}")
            del self.ring[key]
            self.sorted_keys.remove(key)

    def get_node(self, key):
        if not self.ring:
            return None
        hash_key = self._hash(key)
        for ring_key in self.sorted_keys:
            if hash_key <= ring_key:
                return self.ring[ring_key]
        return self.ring[self.sorted_keys[0]]

# Usage
ch = ConsistentHash(["server1", "server2", "server3"])
print(ch.get_node("user:123"))  # -> server2`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-leader-election',
    title: 'Leader Election',
    category: 'Distributed Systems',
    description: 'Choose a single node to coordinate actions in a distributed system. Critical for consensus protocols.',
    keyPoints: [
      'Only one leader at a time (prevents split-brain)',
      'Automatic failover when leader fails',
      'Algorithms: Bully, Raft, Paxos, ZAB',
      'Leases prevent stale leaders',
      'Tools: Zookeeper, etcd, Consul',
    ],
    codeExample: `# Leader Election with Redis (simplified)
import redis
import time
import uuid

class LeaderElection:
    def __init__(self, redis_client, key, ttl=10):
        self.redis = redis_client
        self.key = f"leader:{key}"
        self.ttl = ttl
        self.node_id = str(uuid.uuid4())
        self.is_leader = False

    def try_acquire_leadership(self):
        # SET NX with TTL (atomic)
        acquired = self.redis.set(
            self.key,
            self.node_id,
            nx=True,  # Only if not exists
            ex=self.ttl
        )
        self.is_leader = acquired
        return acquired

    def renew_leadership(self):
        if not self.is_leader:
            return False
        # Only renew if we're still leader
        current = self.redis.get(self.key)
        if current == self.node_id:
            self.redis.expire(self.key, self.ttl)
            return True
        self.is_leader = False
        return False

    def run(self, leader_work, follower_work):
        while True:
            if self.is_leader:
                if self.renew_leadership():
                    leader_work()
                else:
                    follower_work()
            else:
                if self.try_acquire_leadership():
                    leader_work()
                else:
                    follower_work()
            time.sleep(self.ttl / 3)`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-distributed-locking',
    title: 'Distributed Locking',
    category: 'Distributed Systems',
    description: 'Coordinate access to shared resources across multiple nodes to prevent race conditions.',
    keyPoints: [
      'Mutual exclusion across distributed nodes',
      'Must handle node failures and network partitions',
      'Redlock algorithm for Redis-based locks',
      'Lock with TTL to prevent deadlocks',
      'Fencing tokens prevent stale lock holders',
    ],
    codeExample: `# Redis Distributed Lock
class DistributedLock:
    def __init__(self, redis_client, resource, ttl=10):
        self.redis = redis_client
        self.resource = f"lock:{resource}"
        self.ttl = ttl
        self.token = str(uuid.uuid4())

    def acquire(self, blocking=True, timeout=None):
        start = time.time()
        while True:
            if self.redis.set(self.resource, self.token, nx=True, ex=self.ttl):
                return True
            if not blocking:
                return False
            if timeout and (time.time() - start) > timeout:
                return False
            time.sleep(0.1)

    def release(self):
        # Lua script for atomic check-and-delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        return self.redis.eval(lua_script, 1, self.resource, self.token)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

# Usage
lock = DistributedLock(redis, "order:123")
with lock:
    process_order()

# Redlock (multi-node Redis)
# Acquire lock from N/2+1 Redis instances
# All locks must succeed within TTL window`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-vector-clocks',
    title: 'Vector Clocks & Conflict Resolution',
    category: 'Distributed Systems',
    description: 'Track causality in distributed systems to detect conflicts and enable eventual consistency.',
    keyPoints: [
      'Each node maintains a vector of logical clocks',
      'Increment own clock on local events',
      'Merge clocks on message receive',
      'Detect concurrent updates for conflict resolution',
      'Used in: DynamoDB, Riak, Cassandra',
    ],
    codeExample: `# Vector Clock Implementation
class VectorClock:
    def __init__(self, node_id):
        self.node_id = node_id
        self.clock = {}

    def increment(self):
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1
        return self.copy()

    def merge(self, other_clock):
        for node, time in other_clock.clock.items():
            self.clock[node] = max(self.clock.get(node, 0), time)

    def compare(self, other):
        # Returns: 'before', 'after', 'concurrent'
        dominated = True
        dominates = True

        all_nodes = set(self.clock.keys()) | set(other.clock.keys())
        for node in all_nodes:
            my_time = self.clock.get(node, 0)
            other_time = other.clock.get(node, 0)
            if my_time < other_time:
                dominates = False
            if my_time > other_time:
                dominated = False

        if dominated and not dominates:
            return 'before'
        if dominates and not dominated:
            return 'after'
        return 'concurrent'  # Conflict!

# Example: Concurrent writes
# Node A: {A: 1}
# Node B: {B: 1}
# Compare: concurrent -> need conflict resolution

# Conflict Resolution Strategies:
# 1. Last-Write-Wins (LWW) - use timestamps
# 2. Application-level merge (CRDTs)
# 3. Return all versions to client (like Amazon cart)`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-bloom-filters',
    title: 'Bloom Filters',
    category: 'Distributed Systems',
    description: 'Space-efficient probabilistic data structure to test set membership. May have false positives but never false negatives.',
    keyPoints: [
      'Uses k hash functions and m bits',
      'Add: Set k bit positions to 1',
      'Query: Check if all k positions are 1',
      'False positives possible, false negatives impossible',
      'Use cases: Caching, spell check, malware detection',
    ],
    codeExample: `# Bloom Filter Implementation
import mmh3  # MurmurHash

class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = [0] * size

    def _hashes(self, item):
        for i in range(self.num_hashes):
            yield mmh3.hash(item, i) % self.size

    def add(self, item):
        for pos in self._hashes(item):
            self.bits[pos] = 1

    def might_contain(self, item):
        return all(self.bits[pos] for pos in self._hashes(item))

# Usage
bf = BloomFilter(size=1000000, num_hashes=7)

# Add items
bf.add("user:123")
bf.add("user:456")

# Query
bf.might_contain("user:123")  # True (definitely added)
bf.might_contain("user:789")  # False (definitely not added)
bf.might_contain("user:999")  # True? (maybe false positive)

# Use case: Avoid expensive DB lookups
def get_user(user_id):
    if not bloom_filter.might_contain(user_id):
        return None  # Definitely not in DB, skip query
    return db.query_user(user_id)  # Might exist, check DB`,
    codeLanguage: 'python',
    topicType: 'systemDesign',
  },

  // ===== ESTIMATION & CALCULATIONS =====
  {
    id: 'sd-back-of-envelope',
    title: 'Back-of-Envelope Calculations',
    category: 'Estimation',
    description: 'Quick estimations for capacity planning. Know your powers of 2 and latency numbers.',
    keyPoints: [
      'Know latency numbers: L1 cache 1ns, RAM 100ns, SSD 100us, Network 1ms',
      'Powers of 2: 2^10=1K, 2^20=1M, 2^30=1B, 2^40=1T',
      'QPS calculations: Daily users / 86400 seconds',
      'Storage: Size per record * number of records * replication',
      'Bandwidth: QPS * average response size',
    ],
    codeExample: `# Latency Numbers Every Programmer Should Know
L1 cache reference:                  1 ns
L2 cache reference:                  4 ns
Main memory reference:             100 ns
SSD random read:                   100 us
HDD seek:                           10 ms
Network round trip (same DC):      500 us
Network round trip (cross-country): 150 ms

# Powers of 2
2^10 = 1,024           ~ 1 Thousand (KB)
2^20 = 1,048,576       ~ 1 Million (MB)
2^30 = 1,073,741,824   ~ 1 Billion (GB)
2^40 = 1,099,511,627,776 ~ 1 Trillion (TB)

# Example: Twitter-like Service
Daily Active Users: 300 million
Tweets per user per day: 2
Total tweets/day: 600 million
Tweets/second: 600M / 86400 = ~7,000 TPS

Read:Write ratio: 100:1
Read QPS: 700,000

Tweet size: 280 chars * 2 bytes = 560 bytes
Metadata: 200 bytes
Total per tweet: ~800 bytes

Daily storage: 600M * 800B = 480 GB
Yearly storage: 480 GB * 365 = 175 TB
With 3x replication: 525 TB

# Bandwidth
Read bandwidth: 700K QPS * 1KB = 700 MB/s`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-capacity-planning',
    title: 'Capacity Planning',
    category: 'Estimation',
    description: 'Plan infrastructure for expected load with room for growth. Balance cost vs performance.',
    keyPoints: [
      'Start with requirements: Users, QPS, storage, latency',
      'Calculate with peak load (usually 2-3x average)',
      'Plan for 3-5 years of growth',
      'Build in headroom (70-80% utilization target)',
      'Consider geographic distribution',
    ],
    codeExample: `# Capacity Planning Template

# 1. Traffic Estimation
Daily Active Users: 10 million
Actions per user/day: 10
Total actions/day: 100 million
Average QPS: 100M / 86400 = 1,157 QPS
Peak QPS (3x): 3,500 QPS

# 2. Storage Estimation
Record size: 1 KB
Records/day: 100 million
Daily storage: 100 GB
Retention: 5 years
Total storage: 100 GB * 365 * 5 = 182 TB
With replication (3x): 546 TB

# 3. Bandwidth
Read: 3500 QPS * 2KB = 7 MB/s = 56 Mbps
Write: 1200 QPS * 1KB = 1.2 MB/s = 10 Mbps

# 4. Server Estimation
Assume 1 server handles 500 QPS
Servers needed: 3500 / 500 = 7 servers
With redundancy (3x): 21 servers

# 5. Database Sizing
If single machine handles 10K QPS read, 1K write
Read replicas: 3500 / 10000 = 1 (min 3 for HA)
Write capacity: 1200 / 1000 = 2 masters

# 6. Cache Sizing
Cache hit ratio target: 90%
Working set: 10 million users * 1KB = 10 GB
With overhead: 15 GB per cache node
Redis nodes: 3 (for HA)`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },

  // ===== DESIGN INTERVIEW TIPS =====
  {
    id: 'sd-interview-framework',
    title: 'System Design Interview Framework',
    category: 'Interview Tips',
    description: 'A structured approach to tackle system design interviews confidently.',
    keyPoints: [
      '1. Requirements: Functional and non-functional (5 min)',
      '2. Estimation: Traffic, storage, bandwidth (5 min)',
      '3. High-level design: Core components (10 min)',
      '4. Detailed design: Deep dive into 2-3 components (15 min)',
      '5. Bottlenecks: Scaling, failure handling (5 min)',
    ],
    codeExample: `# System Design Interview Template

## 1. Requirements Gathering (5 min)
Functional:
- What features are in scope?
- Who are the users?
- What are the main use cases?

Non-functional:
- Scale: How many users? QPS?
- Latency: What's acceptable response time?
- Availability: What uptime is required?
- Consistency: Strong or eventual?

## 2. Capacity Estimation (5 min)
- QPS (read and write)
- Storage requirements
- Bandwidth needs
- Cache sizing

## 3. High-Level Design (10 min)
- Draw main components
- Show data flow
- Identify databases/caches
- API design (key endpoints)

## 4. Detailed Design (15 min)
- Deep dive into critical components
- Database schema
- Algorithm choices
- Trade-off discussions

## 5. Scale & Reliability (5 min)
- Identify bottlenecks
- Add load balancers, caching
- Database sharding/replication
- Failure scenarios and handling

# Pro Tips:
- Drive the conversation
- Think out loud
- Justify decisions
- Acknowledge trade-offs`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
  {
    id: 'sd-trade-offs',
    title: 'Common Trade-offs',
    category: 'Interview Tips',
    description: 'Key trade-offs to discuss in system design interviews. There are no perfect solutions.',
    keyPoints: [
      'Consistency vs Availability (CAP theorem)',
      'Latency vs Throughput',
      'Storage vs Computation',
      'Complexity vs Maintainability',
      'Cost vs Performance',
    ],
    codeExample: `# Key Trade-offs in System Design

# 1. Consistency vs Availability
Strong Consistency: Banking, inventory
- Every read gets latest write
- May be unavailable during partitions

Eventual Consistency: Social media, caching
- Reads may be stale
- Always available

# 2. Latency vs Throughput
Low Latency: Real-time gaming, trading
- In-memory caching, edge servers
- May sacrifice throughput

High Throughput: Batch processing, analytics
- Queue requests, batch operations
- May increase latency

# 3. SQL vs NoSQL
SQL: Complex queries, ACID transactions
- Joins, relationships
- Vertical scaling limits

NoSQL: Flexible schema, horizontal scale
- Denormalized data
- Eventually consistent

# 4. Push vs Pull
Push (Fan-out on write): Fast reads, high write cost
- Good for normal users
- Wasted work if not read

Pull (Fan-out on read): Fast writes, slow reads
- Good for celebrities
- Read cost on demand

# 5. Synchronous vs Asynchronous
Sync: Simple, immediate feedback
- Blocking, coupled
- Harder to scale

Async: Decoupled, scalable
- Complex, eventual
- Message queues needed`,
    codeLanguage: 'plaintext',
    topicType: 'systemDesign',
  },
];
