-- ============================================================================
-- A.D.R.I.A.N Database Schema Initialization
-- PostgreSQL Database Tables for Memory, Users, Conversations, and Preferences
-- ============================================================================

-- Enable UUID extension for generating unique IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USERS TABLE
-- Stores user authentication info, profiles, and voice biometrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    voice_profile BYTEA,  -- Stored voice biometric data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB  -- Additional user metadata
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active);

-- ============================================================================
-- SESSIONS TABLE
-- Track user interaction sessions
-- ============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    session_metadata JSONB,  -- Store session context
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_active ON sessions(is_active);
CREATE INDEX idx_sessions_start_time ON sessions(start_time DESC);

-- ============================================================================
-- CONVERSATIONS TABLE
-- Store all conversation history with intent and context
-- ============================================================================
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    message_type VARCHAR(50) NOT NULL,  -- 'user', 'assistant', 'system'
    intent VARCHAR(100),  -- Classified intent
    confidence FLOAT,  -- Intent classification confidence
    correlation_id VARCHAR(100),  -- For tracking across services
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB  -- Additional context (entities, sentiment, etc.)
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_session_id ON conversations(session_id);
CREATE INDEX idx_conversations_timestamp ON conversations(timestamp DESC);
CREATE INDEX idx_conversations_intent ON conversations(intent);
CREATE INDEX idx_conversations_correlation_id ON conversations(correlation_id);

-- ============================================================================
-- MEMORY_EMBEDDINGS TABLE
-- Metadata for FAISS vector embeddings (actual vectors stored in FAISS)
-- ============================================================================
CREATE TABLE IF NOT EXISTS memory_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    text TEXT NOT NULL,  -- Original text that was embedded
    embedding_index INTEGER NOT NULL,  -- Index in FAISS vector store
    source_type VARCHAR(50),  -- 'conversation', 'note', 'document', etc.
    source_id UUID,  -- Reference to source (e.g., conversation_id)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB  -- Additional metadata for semantic search
);

CREATE INDEX idx_memory_embeddings_user_id ON memory_embeddings(user_id);
CREATE INDEX idx_memory_embeddings_index ON memory_embeddings(embedding_index);
CREATE INDEX idx_memory_embeddings_source ON memory_embeddings(source_type, source_id);
CREATE INDEX idx_memory_embeddings_created_at ON memory_embeddings(created_at DESC);

-- ============================================================================
-- PREFERENCES TABLE
-- Store user preferences and learned behaviors (key-value store)
-- ============================================================================
CREATE TABLE IF NOT EXISTS preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    category VARCHAR(50) NOT NULL,  -- 'ui', 'behavior', 'personality', etc.
    key VARCHAR(100) NOT NULL,
    value JSONB NOT NULL,  -- Flexible value storage
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, category, key)
);

CREATE INDEX idx_preferences_user_id ON preferences(user_id);
CREATE INDEX idx_preferences_category ON preferences(category);
CREATE INDEX idx_preferences_key ON preferences(key);

-- ============================================================================
-- SYSTEM_LOGS TABLE
-- Store important system events and actions for audit/debugging
-- ============================================================================
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,  -- 'action_executed', 'error', 'security_check', etc.
    event_data JSONB NOT NULL,
    severity VARCHAR(20) DEFAULT 'info',  -- 'debug', 'info', 'warning', 'error', 'critical'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_system_logs_user_id ON system_logs(user_id);
CREATE INDEX idx_system_logs_event_type ON system_logs(event_type);
CREATE INDEX idx_system_logs_severity ON system_logs(severity);
CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp DESC);

-- ============================================================================
-- TASKS TABLE
-- Store scheduled tasks, reminders, and automation
-- ============================================================================
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    task_type VARCHAR(50),  -- 'reminder', 'automation', 'scheduled_action'
    scheduled_time TIMESTAMP,
    is_completed BOOLEAN DEFAULT FALSE,
    is_recurring BOOLEAN DEFAULT FALSE,
    recurrence_rule VARCHAR(255),  -- Cron-like expression
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_scheduled_time ON tasks(scheduled_time);
CREATE INDEX idx_tasks_completed ON tasks(is_completed);
CREATE INDEX idx_tasks_type ON tasks(task_type);

-- ============================================================================
-- Insert Default User
-- ============================================================================
INSERT INTO users (username, email, metadata)
VALUES ('default_user', 'default@adrian.local', '{"role": "primary", "created_by": "system"}')
ON CONFLICT (username) DO NOTHING;

-- ============================================================================
-- Update Timestamp Function
-- Automatically update 'updated_at' column on row updates
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables with updated_at column
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_preferences_updated_at
    BEFORE UPDATE ON preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- Recent conversations with user info
CREATE OR REPLACE VIEW v_recent_conversations AS
SELECT 
    c.id,
    c.user_id,
    u.username,
    c.message,
    c.message_type,
    c.intent,
    c.timestamp,
    c.metadata
FROM conversations c
JOIN users u ON c.user_id = u.id
ORDER BY c.timestamp DESC;

-- Active sessions with user info
CREATE OR REPLACE VIEW v_active_sessions AS
SELECT 
    s.id as session_id,
    s.user_id,
    u.username,
    s.start_time,
    s.session_metadata,
    COUNT(c.id) as message_count
FROM sessions s
JOIN users u ON s.user_id = u.id
LEFT JOIN conversations c ON c.session_id = s.id
WHERE s.is_active = TRUE
GROUP BY s.id, s.user_id, u.username, s.start_time, s.session_metadata;

-- ============================================================================
-- Database Initialization Complete
-- ============================================================================

