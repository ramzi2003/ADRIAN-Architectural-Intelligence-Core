# ADRIAN Data Flow & Message Structure

## Overview

This document describes how data flows through the ADRIAN microservices architecture and the structure of messages exchanged between services.

---

## Message Channels (Redis Pub/Sub)

| Channel | Publisher | Subscriber(s) | Message Type |
|---------|-----------|---------------|--------------|
| `utterances` | io-service | processing-service | UtteranceEvent |
| `action_specs` | processing-service | execution-service | ActionSpec |
| `action_results` | execution-service | output-service | ActionResult |
| `responses` | processing-service | output-service | ResponseText |
| `security_checks` | processing-service | security-service | SecurityCheck |
| `security_results` | security-service | execution-service | SecurityResult |

---

## Data Flow Sequences

### 1. Simple Conversation Flow

```
User speaks/types → IO Service
                    ↓
          [UtteranceEvent via Redis]
                    ↓
              Processing Service
                    ↓ (queries memory)
              Memory Service
                    ↓ (returns context)
              Processing Service
                    ↓ (generates response)
          [ResponseText via Redis]
                    ↓
               Output Service
                    ↓
          Speaks response to user
```

### 2. Action Execution Flow

```
User: "open chrome" → IO Service
                       ↓
             [UtteranceEvent via Redis]
                       ↓
                 Processing Service
                   ↓ (classifies intent)
                 Intent: "open_app"
                   ↓ (creates action_spec)
             [ActionSpec via Redis]
                       ↓
                 Execution Service
                   ↓ (performs action)
             [ActionResult via Redis]
                       ↓
                 Output Service
                   ↓ (confirms action)
               TTS: "At once, Sir."
```

### 3. Permission-Required Action Flow

```
User: "delete important_file.txt" → IO Service
                                     ↓
                       [UtteranceEvent via Redis]
                                     ↓
                               Processing Service
                                 ↓ (detects dangerous intent)
                       [SecurityCheck via Redis]
                                     ↓
                               Security Service
                                 ↓ (verifies permission)
                       [SecurityResult: approved=false]
                                     ↓
                               Processing Service
                                 ↓ (generates denial response)
                       [ResponseText via Redis]
                                     ↓
                               Output Service
                                     ↓
                    TTS: "Sir, I cannot delete that without explicit confirmation."
```

### 4. Memory Storage Flow

```
User conversation → Processing Service
                    ↓ (after generating response)
              [Store in Memory Service]
                    ↓ (HTTP POST)
              Memory Service
                    ↓ (generates embedding)
              Sentence Transformer
                    ↓
         Stores in PostgreSQL + FAISS index
                    ↓
              Learning loop complete
```

---

## Message Schemas

### UtteranceEvent

```json
{
  "message_id": "uuid",
  "correlation_id": "uuid",
  "timestamp": "2025-10-07T12:00:00Z",
  "message_type": "utterance",
  "text": "open chrome",
  "confidence": 1.0,
  "source": "voice",  // or "text"
  "user_id": "default_user"
}
```

### ActionSpec

```json
{
  "message_id": "uuid",
  "correlation_id": "uuid",
  "timestamp": "2025-10-07T12:00:00Z",
  "message_type": "action_spec",
  "intent": "open_app",
  "parameters": {
    "app_name": "chrome"
  },
  "requires_permission": false,
  "confidence": 0.95
}
```

### ActionResult

```json
{
  "message_id": "uuid",
  "correlation_id": "uuid",
  "timestamp": "2025-10-07T12:00:01Z",
  "message_type": "action_result",
  "action_id": "original_action_spec_id",
  "success": true,
  "result_data": {
    "message": "Chrome opened successfully"
  },
  "error_message": null
}
```

### ResponseText

```json
{
  "message_id": "uuid",
  "correlation_id": "uuid",
  "timestamp": "2025-10-07T12:00:00Z",
  "message_type": "response_text",
  "text": "At once, Sir. Opening Chrome.",
  "should_speak": true,
  "emotion": "neutral"
}
```

### SecurityCheck

```json
{
  "message_id": "uuid",
  "correlation_id": "uuid",
  "timestamp": "2025-10-07T12:00:00Z",
  "message_type": "security_check",
  "action_intent": "delete_file",
  "user_id": "default_user",
  "requires_auth": true
}
```

### SecurityResult

```json
{
  "message_id": "uuid",
  "correlation_id": "uuid",
  "timestamp": "2025-10-07T12:00:00Z",
  "message_type": "security_result",
  "check_id": "original_security_check_id",
  "approved": false,
  "reason": "Dangerous action requires explicit confirmation"
}
```

---

## Correlation IDs

Every request flow uses a **correlation_id** that is:
- Generated by the IO Service when receiving input
- Propagated through all messages in the flow
- Used for tracing and debugging across services
- Logged in the SQLite database for analysis

**Example trace query:**

```sql
SELECT service_name, level, message, timestamp
FROM logs
WHERE correlation_id = 'abc-123-def-456'
ORDER BY timestamp;
```

---

## Synchronous API Calls

Some operations use synchronous HTTP calls instead of pub/sub:

### Memory Service Queries

```http
POST /memory/search
{
  "query": "what did I ask about yesterday?",
  "limit": 5,
  "user_id": "default_user"
}
```

### Security Permission Checks

```http
POST /auth/check-permission
{
  "user_id": "default_user",
  "action_intent": "open_app"
}
```

### Direct Action Execution (Testing)

```http
POST /execute
{
  "intent": "open_app",
  "parameters": {"app_name": "chrome"}
}
```

---

## Error Handling

### Service Unavailable

If a service is down:
- Publisher logs error but continues
- Subscriber timeout handled gracefully
- Error propagated via ActionResult with `success: false`

### Message Validation Errors

- Invalid message schema → logged and discarded
- Missing required fields → error response generated
- Type mismatches → validation error returned

### Redis Connection Loss

- Services attempt reconnection with exponential backoff
- Queued messages lost (pub/sub is fire-and-forget)
- Critical data persisted in PostgreSQL before publishing

---

## Performance Considerations

| Operation | Latency Target | Notes |
|-----------|---------------|-------|
| Redis publish | <1ms | Local Redis is very fast |
| Message processing | <100ms | Per service hop |
| Memory semantic search | <200ms | FAISS index search |
| Ollama LLM inference | 1-5s | Model-dependent |
| Gemini API call | 500ms-2s | Network-dependent |
| End-to-end response | <2s | For simple commands |

---

## Testing Message Flow

### Using cURL

```bash
# Send utterance
curl -X POST http://localhost:8001/input/text \
  -H "Content-Type: application/json" \
  -d '{"text": "open chrome", "user_id": "test_user"}'
```

### Using PowerShell

```powershell
$body = @{
    text = "open chrome"
    user_id = "test_user"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8001/input/text `
  -Method POST -Body $body -ContentType "application/json"
```

---

## Debugging Tips

1. **Follow correlation IDs**: Track a request through all services
2. **Check Redis**: Use `redis-cli MONITOR` to see all pub/sub traffic
3. **Query logs**: SQLite database has all events with timestamps
4. **Service health**: Check `/health` endpoints on all services
5. **Individual testing**: Test each service independently via REST API

---

**Status**: Task 3 Complete - Message flow documented and ready for Phase 2 implementation.

