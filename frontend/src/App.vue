<template>
  <div class="app-layout">
    <Sidebar
      v-model="mode"
      v-model:customRole="customRole"
      @clear-history="clearHistory"
      @mode-change="fetchGreeting"
    />

    <PerceptionPanel
      ref="perceptionRef"
      :loading="loading"
      :chartData="chartData"
      @request-stt="handleSTT"
      @confirm-analysis="handleAnalysis"
    />

    <ChatWindow
      :messages="messages"
      :currentEmotion="currentEmotion"
    />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import Sidebar from './components/Sidebar.vue'
import ChatWindow from './components/ChatWindow.vue'
import PerceptionPanel from './components/PerceptionPanel.vue'

// --- 状态 ---
const mode = ref('日常闲聊')
const customRole = ref('')
const loading = ref(false)
const currentEmotion = ref('NEUTRAL')
const messages = ref([])
const chartData = ref(null)
const perceptionRef = ref(null)

const API_BASE = 'http://127.0.0.1:8000/api'

// --- 初始化 ---
onMounted(() => {
  fetchGreeting()
})

// --- 业务逻辑 ---

// 1. 处理 STT 请求
async function handleSTT(audioBlob) {
  loading.value = true
  const formData = new FormData()
  formData.append('audio', audioBlob, 'temp.wav')

  try {
    const res = await axios.post(`${API_BASE}/stt`, formData)
    if (res.data.status === 'success') {
      perceptionRef.value.setDraftText(res.data.text)
    } else {
      ElMessage.error('识别失败: ' + res.data.message)
    }
  } catch (err) {
    ElMessage.error('网络错误')
  } finally {
    loading.value = false
  }
}

// 2. 处理最终分析
async function handleAnalysis({ audio, image, text }) {
  loading.value = true
  const formData = new FormData()
  formData.append('audio', audio, 'record.wav')
  if (image) formData.append('image', image, 'capture.jpg')
  formData.append('text', text)
  formData.append('mode', mode.value)
  formData.append('custom_role', customRole.value)

  // 过滤空消息
  const historyPayload = messages.value
    .slice(-10)
    .filter(m => m.content && m.content.trim() !== '')
    .map(m => ({ role: m.role, content: m.content }))

  formData.append('history', JSON.stringify(historyPayload))

  try {
    const res = await axios.post(`${API_BASE}/analyze`, formData)
    if (res.data.status === 'success') {
      const data = res.data.data
      currentEmotion.value = data.emotion
      chartData.value = data

      messages.value.push({
        role: 'user',
        content: data.text,
        emotion: data.emotion,
        confidence: data.confidence
      })
      messages.value.push({ role: 'assistant', content: data.reply })
    }
  } catch (err) {
    ElMessage.error('分析失败')
  } finally {
    loading.value = false
  }
}

// 3. 获取开场白
async function fetchGreeting() {
  loading.value = true
  messages.value = [] // 清空历史
  try {
    const res = await axios.post(`${API_BASE}/greeting`, { mode: mode.value, custom_role: customRole.value })
    if (res.data.status === 'success') {
      messages.value.push({ role: 'assistant', content: res.data.reply })
    }
  } catch (err) {
    messages.value.push({ role: 'assistant', content: `你好，我是${mode.value}助手。` })
  } finally {
    loading.value = false
  }
}

function clearHistory() {
  fetchGreeting()
}
</script>

<style>
/* 全局布局 */
.app-layout { display: flex; height: 100vh; width: 100vw; background-color: #141414; color: #e0e0e0; overflow: hidden; font-family: 'Inter', sans-serif; }
body { margin: 0; }
</style>