<template>
  <aside class="col-right">
    <div class="chat-header">
      <div class="status-pill" :style="{ borderColor: getEmotionColor(currentEmotion), boxShadow: `0 0 15px ${getEmotionColor(currentEmotion)}20` }">
        <span class="status-dot" :style="{ backgroundColor: getEmotionColor(currentEmotion) }"></span>
        <span class="status-text" :style="{ color: getEmotionColor(currentEmotion) }">{{ currentEmotion }} DETECTED</span>
      </div>
    </div>

    <div class="chat-viewport" ref="chatRef">
      <div v-if="messages.length === 0" class="empty-state">
        <div class="empty-icon">💬</div>
        <p>Awaiting Input...</p>
      </div>

      <div v-for="(msg, index) in messages" :key="index" class="msg-row" :class="msg.role">
        <div class="avatar">
          <img v-if="msg.role === 'assistant'" src="https://api.iconify.design/fluent-emoji:robot.svg" />
          <img v-else src="https://api.iconify.design/fluent-emoji:person-raising-hand-medium-light.svg" />
        </div>

        <div class="bubble-wrapper">
          <div v-if="msg.role === 'user'" class="meta-info">
            <span class="emo-tag" :style="{ color: getEmotionColor(msg.emotion), borderColor: getEmotionColor(msg.emotion) }">
              {{ msg.emotion }}
            </span>
            <span v-if="msg.confidence" class="conf-tag">
              {{ (msg.confidence * 100).toFixed(1) }}%
            </span>
          </div>

          <div class="bubble">{{ msg.content }}</div>
        </div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps(['messages', 'currentEmotion'])
const chatRef = ref(null)

watch(() => props.messages, () => {
  nextTick(() => {
    if (chatRef.value) {
      chatRef.value.scrollTo({ top: chatRef.value.scrollHeight, behavior: 'smooth' })
    }
  })
}, { deep: true })

function getEmotionColor(emo) {
  const map = { 'ANGRY': '#FF4D4F', 'HAPPY': '#52C41A', 'SAD': '#1890FF', 'NEUTRAL': '#8c8c8c', 'FEAR': '#722ED1', 'SURPRISE': '#FAAD14', 'DISGUST': '#13C2C2' }
  return map[emo] || '#8c8c8c'
}
</script>

<style scoped>
.col-right { flex: 6; background: #141414; display: flex; flex-direction: column; border-left: 1px solid #2b2b30; }

/* Header */
.chat-header { height: 64px; border-bottom: 1px solid #2b2b30; display: flex; align-items: center; justify-content: center; background: rgba(20,20,20,0.8); backdrop-filter: blur(10px); z-index: 10; }
.status-pill { padding: 6px 16px; border-radius: 100px; border: 1px solid; display: flex; align-items: center; gap: 8px; background: rgba(0,0,0,0.3); transition: all 0.3s; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 10px currentColor; animation: pulse 2s infinite; }
.status-text { font-size: 12px; font-weight: 800; letter-spacing: 1px; }

/* Viewport */
.chat-viewport { flex: 1; overflow-y: auto; padding: 20px 40px; scroll-behavior: smooth; }
.empty-state { height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #333; gap: 10px; opacity: 0.5; }
.empty-icon { font-size: 40px; filter: grayscale(1); }

/* Messages */
.msg-row { display: flex; margin-bottom: 30px; animation: slideUp 0.3s ease; }
.msg-row.user { flex-direction: row-reverse; }

.avatar { width: 42px; height: 42px; margin: 0 16px; border-radius: 12px; background: #232328; display: flex; align-items: center; justify-content: center; border: 1px solid #333; flex-shrink: 0; }
.avatar img { width: 70%; height: 70%; }

.bubble-wrapper { max-width: 75%; display: flex; flex-direction: column; }
.user .bubble-wrapper { align-items: flex-end; }

/* Metadata Tags */
.meta-info { display: flex; gap: 8px; margin-bottom: 6px; align-items: center; }
.emo-tag { font-size: 10px; font-weight: 800; padding: 2px 6px; border: 1px solid; border-radius: 4px; text-transform: uppercase; }
.conf-tag { font-size: 10px; color: #666; font-family: monospace; }

.bubble { padding: 14px 18px; border-radius: 16px; font-size: 15px; line-height: 1.6; color: #eee; position: relative; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.user .bubble { background: linear-gradient(135deg, #177ddc, #096dd9); border-top-right-radius: 2px; border: 1px solid #177ddc; }
.assistant .bubble { background: #232328; border-top-left-radius: 2px; border: 1px solid #333; }

@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
@keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>