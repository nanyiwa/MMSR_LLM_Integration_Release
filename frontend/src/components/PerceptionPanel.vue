<template>
  <main class="col-middle">
    <div class="perception-panel">
      <div class="panel-header"><el-icon><View /></el-icon> <span>全息感知终端</span></div>

      <div class="camera-wrapper">
        <video ref="videoPlayer" autoplay muted playsinline class="video-feed"></video>
        <canvas ref="overlayCanvas" class="camera-overlay-canvas"></canvas>
        <div class="camera-overlay">
          <div class="status-indicator" :class="{ active: isCameraOn }">
            <span class="blink-dot"></span> {{ isCameraOn ? 'VISION ONLINE' : 'OFFLINE' }}
          </div>
        </div>
      </div>

      <div class="control-zone">
        <div v-if="workflowState === 'IDLE' || workflowState === 'RECORDING'" class="mic-wrapper">
          <button
            class="mic-button"
            :class="{ recording: workflowState === 'RECORDING', processing: loading }"
            @mousedown="startRecording"
            @mouseup="stopRecording"
          >
            <el-icon v-if="!loading" size="32"><Microphone /></el-icon>
            <el-icon v-else class="is-loading" size="32"><Loading /></el-icon>
          </button>
          <div class="mic-label">
            {{ loading ? '多模态计算中...' : (workflowState === 'RECORDING' ? '松开手指发送' : '按住说话') }}
          </div>
          </div>

        <div v-if="workflowState === 'EDITING'" class="edit-wrapper">
          <div class="editor-container">
            <div class="editor-header">
              <span class="editor-label">语音识别结果</span>
              <span class="editor-hint">点击文字可修改</span>
            </div>
            <el-input
              v-model="localDraftText"
              type="textarea"
              :rows="3"
              class="stt-editor-pro"
              resize="none"
            />
          </div>
          <div class="edit-actions">
            <el-button type="info" circle size="large" @click="cancelEdit" class="action-btn"><el-icon><Close /></el-icon></el-button>
            <el-button type="success" round size="large" @click="confirmSend" class="action-btn send-btn">
              发送 <el-icon class="el-icon--right"><Select /></el-icon>
            </el-button>
          </div>
        </div>
      </div>
    </div>

    <div class="data-panel">
      <div class="data-toolbar">
        <el-tabs v-model="activeTab" class="custom-tabs">
          <el-tab-pane label="融合" name="fusion"></el-tab-pane>
          <el-tab-pane label="听觉" name="audio"></el-tab-pane>
          <el-tab-pane label="视觉" name="vision"></el-tab-pane>
          <el-tab-pane label="文本" name="text"></el-tab-pane>
        </el-tabs>

        <div v-if="activeTab === 'vision'" class="vision-switch">
          <el-switch
            v-model="showSnapshot"
            active-text="快照"
            inactive-text="图表"
            inline-prompt
            style="--el-switch-on-color: #409eff; --el-switch-off-color: #606266"
          />
        </div>
      </div>

      <div class="data-content">
        <div class="chart-wrapper">
          <div v-show="activeTab === 'fusion'" ref="chartFusion" class="chart-instance"></div>
          <div v-show="activeTab === 'audio'" ref="chartAudio" class="chart-instance"></div>
          <div v-show="activeTab === 'text'" ref="chartText" class="chart-instance"></div>

          <div v-show="activeTab === 'vision'" class="vision-container">
            <div v-show="!showSnapshot" ref="chartVision" class="chart-instance"></div>

            <div v-if="showSnapshot" class="snapshot-viewer">
              <div v-if="decisionImageUrl" class="img-box">
                <img :src="decisionImageUrl" />
                <div class="img-tag">Decision Frame</div>
              </div>
              <div v-else class="no-data">
                <el-icon :size="30"><Picture /></el-icon>
                <p>暂无捕获数据</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'

const props = defineProps(['loading', 'chartData'])
const emit = defineEmits(['request-stt', 'confirm-analysis'])

// 状态
const isCameraOn = ref(false)
const workflowState = ref('IDLE')
const localDraftText = ref('')
const activeTab = ref('fusion')
const showSnapshot = ref(false)
const cachedBlobs = ref({ audio: null, image: null })
const decisionImageUrl = ref('')

// 抓拍相关
const bestFrameBlob = ref(null)
const bestFrameScore = ref(0)

const videoPlayer = ref(null)
const mediaRecorder = ref(null)
const overlayCanvas = ref(null)
const audioChunks = ref([])
let liveVisionTimer = null

const chartFusion = ref(null), chartAudio = ref(null), chartVision = ref(null), chartText = ref(null)
let chartInstances = {}

const setDraftText = (text) => { localDraftText.value = text; workflowState.value = 'EDITING' }
defineExpose({ setDraftText })

// ==========================
// 实时视觉检测
// ==========================
function startLiveVision() {
  liveVisionTimer = setInterval(async () => {
    if (!isCameraOn.value || !videoPlayer.value) return
    try {
      const imageBlob = await captureImage()
      const formData = new FormData()
      formData.append('image', imageBlob)

      const res = await axios.post('http://127.0.0.1:8000/api/live_vision', formData)

      if (res.data.status === 'success') {
        drawFaceBox(res.data.data)
        if (workflowState.value === 'RECORDING') {
          bestFrameBlob.value = imageBlob
          bestFrameScore.value = 0.99
        }
      } else {
        clearCanvas()
      }
    } catch (e) { }
  }, 300)
}

function drawFaceBox(data) {
  const canvas = overlayCanvas.value
  const video = videoPlayer.value
  if (!canvas || !video) return
  const ctx = canvas.getContext('2d')
  if (canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight }

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  const { region, emotion } = data

  ctx.lineWidth = 3
  ctx.strokeStyle = getEmotionColorHex(emotion.toUpperCase())
  ctx.strokeRect(region.x, region.y, region.w, region.h)

  ctx.font = "bold 16px Arial"
  ctx.fillStyle = getEmotionColorHex(emotion.toUpperCase())
  ctx.fillText(emotion.toUpperCase(), region.x, region.y - 10)
}

function clearCanvas() {
  const canvas = overlayCanvas.value
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
}

// ==========================
// 录音控制
// ==========================
function startRecording() {
  if (!mediaRecorder.value) return
  workflowState.value = 'RECORDING'
  audioChunks.value = []
  bestFrameBlob.value = null
  bestFrameScore.value = 0
  mediaRecorder.value.start()
}

function stopRecording() {
  workflowState.value = 'IDLE'
  mediaRecorder.value.stop()
}

// 初始化
onMounted(async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    videoPlayer.value.srcObject = stream
    isCameraOn.value = true
    startLiveVision()

    mediaRecorder.value = new MediaRecorder(stream)
    mediaRecorder.value.ondataavailable = e => { if(e.data.size > 0) audioChunks.value.push(e.data) }
    mediaRecorder.value.onstop = async () => {
      const audioBlob = new Blob(audioChunks.value, { type: 'audio/wav' })

      let finalImageBlob = bestFrameBlob.value
      if (!finalImageBlob) {
        finalImageBlob = await captureImage()
      }

      if (decisionImageUrl.value) URL.revokeObjectURL(decisionImageUrl.value)
      decisionImageUrl.value = URL.createObjectURL(finalImageBlob)

      cachedBlobs.value = { audio: audioBlob, image: finalImageBlob }
      emit('request-stt', audioBlob)
    }

    setTimeout(() => initCharts(), 500)
    window.addEventListener('resize', () => Object.values(chartInstances).forEach(c => c.resize()))
  } catch (err) { ElMessage.error('设备初始化失败') }
})

function confirmSend() {
  emit('confirm-analysis', {
    audio: cachedBlobs.value.audio,
    image: cachedBlobs.value.image,
    text: localDraftText.value
  })
  workflowState.value = 'IDLE'
  localDraftText.value = ''
}

function cancelEdit() {
  workflowState.value = 'IDLE'
  localDraftText.value = ''
}

function captureImage() {
  const canvas = document.createElement('canvas')
  canvas.width = videoPlayer.value.videoWidth
  canvas.height = videoPlayer.value.videoHeight
  canvas.getContext('2d').drawImage(videoPlayer.value, 0, 0)
  return new Promise(r => canvas.toBlob(b => r(b), 'image/jpeg'))
}

// ==========================
// 图表逻辑
// ==========================
function initCharts() {
  initEChart('fusion', chartFusion.value)
  initEChart('audio', chartAudio.value)
  initEChart('vision', chartVision.value)
  initEChart('text', chartText.value)
}

function initEChart(key, dom) {
  if (!dom) return
  const chart = echarts.init(dom)
  chart.setOption({
    grid: { top: 20, bottom: 20, left: 10, right: 30, containLabel: true },
    xAxis: { show: false, max: 1.0 },
    yAxis: { type: 'category', axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: '#bbb', fontWeight: 'bold', fontSize: 12 } },
    series: [{ type: 'bar', barWidth: 24, itemStyle: { borderRadius: [0, 4, 4, 0] }, label: { show: true, position: 'right', color: '#fff', formatter: '{c}', fontWeight: 'bold' } }]
  })
  chartInstances[key] = chart
}

function updateChartData(key, dataObj) {
  const chart = chartInstances[key]
  if (!chart || !dataObj) return
  const sorted = Object.entries(dataObj).sort((a, b) => a[1] - b[1])
  const categories = sorted.map(e => e[0].toUpperCase())
  const values = sorted.map(e => {
    const color = getEmotionColorHex(e[0].toUpperCase())
    return {
      value: e[1].toFixed(2),
      itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: adjustAlpha(color, 0.3) }, { offset: 1, color: color }]) }
    }
  })
  chart.setOption({ yAxis: { data: categories }, series: [{ data: values }] })
}

function adjustAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function getEmotionColorHex(emo) {
  const map = { 'ANGRY': '#FF4D4F', 'HAPPY': '#52C41A', 'SAD': '#1890FF', 'NEUTRAL': '#8c8c8c', 'FEAR': '#722ED1', 'SURPRISE': '#FAAD14', 'DISGUST': '#13C2C2' }
  return map[emo] || '#8c8c8c'
}

// 监听数据变化
watch(() => props.chartData, (newVal) => {
  if (!newVal) return
  updateChartData('fusion', newVal.scores)
  updateChartData('audio', newVal.audio_score)
  updateChartData('vision', newVal.vision_score)
  updateChartData('text', newVal.text_score)
}, { deep: true })

watch([activeTab, showSnapshot], () => {
  nextTick(() => {
    Object.values(chartInstances).forEach(c => c && c.resize())
  })
})

onUnmounted(() => { if (liveVisionTimer) clearInterval(liveVisionTimer) })
</script>

<style scoped>
/* 布局 */
.col-middle { flex: 4; display: flex; flex-direction: column; padding: 20px; background: #141414; min-width: 350px; border-right: 1px solid #2b2b30; }
.perception-panel { flex: 0 0 auto; margin-bottom: 20px; }
.panel-header { font-size: 13px; color: #666; margin-bottom: 12px; display: flex; align-items: center; gap: 6px; letter-spacing: 1px; font-weight: 700; text-transform: uppercase; }

.camera-wrapper { position: relative; width: 100%; aspect-ratio: 16/9; background: #000; border-radius: 12px; overflow: hidden; border: 1px solid #333; box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
.video-feed { width: 100%; height: 100%; object-fit: cover; }
.camera-overlay-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
.camera-overlay { position: absolute; top: 12px; left: 12px; }
.status-indicator { background: rgba(0,0,0,0.6); backdrop-filter: blur(4px); padding: 4px 10px; border-radius: 20px; font-size: 10px; color: #ccc; display: flex; align-items: center; gap: 6px; border: 1px solid rgba(255,255,255,0.1); }
.status-indicator.active { color: #52c41a; border-color: #52c41a; }
.blink-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor; }

.control-zone { margin-top: 24px; min-height: 120px; display: flex; align-items: center; justify-content: center; position: relative; }
.mic-wrapper { text-align: center; }
.mic-button { width: 72px; height: 72px; border-radius: 50%; background: #232328; border: 1px solid #333; color: #fff; cursor: pointer; transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
.mic-button:active, .mic-button.recording { transform: scale(1.1); background: #ff4d4f; border-color: #ff4d4f; box-shadow: 0 0 30px rgba(255, 77, 79, 0.5); }
.mic-button.processing { background: #1890ff; border-color: #1890ff; animation: pulse 1.5s infinite; }
.mic-label { margin-top: 16px; font-size: 12px; color: #666; font-weight: 500; }
.best-frame-tip { position: absolute; bottom: -20px; font-size: 10px; color: #52c41a; animation: fadeInUp 0.3s; }

/* 编辑器美化 */
.edit-wrapper { width: 100%; display: flex; flex-direction: column; align-items: center; gap: 15px; animation: fadeInUp 0.3s; }
.editor-container { width: 100%; background: #232328; border-radius: 12px; border: 1px solid #333; padding: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
.editor-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
.editor-label { font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; }
.editor-hint { font-size: 10px; color: #555; }

:deep(.stt-editor-pro .el-textarea__inner) {
  background: transparent; border: none; color: #fff; font-size: 15px; padding: 0; line-height: 1.6; box-shadow: none;
}
:deep(.stt-editor-pro .el-textarea__inner:focus) { box-shadow: none; }

.edit-actions { display: flex; gap: 20px; align-items: center; }
.action-btn { transition: transform 0.2s; }
.action-btn:hover { transform: scale(1.05); }
.send-btn { font-weight: 700; padding: 20px 30px; letter-spacing: 1px; }

/* 数据面板 */
.data-panel { flex: 1; background: #1f1f24; border-radius: 16px; border: 1px solid #2b2b30; overflow: hidden; display: flex; flex-direction: column; }
.data-toolbar { display: flex; justify-content: space-between; align-items: center; padding: 0 16px; border-bottom: 1px solid #2b2b30; background: #26262b; height: 48px; }
.vision-switch { display: flex; align-items: center; }

.custom-tabs :deep(.el-tabs__header) { border: none; margin: 0; background: transparent; }
.custom-tabs :deep(.el-tabs__item) { color: #888; font-size: 13px; font-weight: 600; height: 48px; line-height: 48px; border: none !important; }
.custom-tabs :deep(.el-tabs__item.is-active) { color: #fff; background: transparent; border-bottom: 2px solid #409eff !important; }

.data-content { flex: 1; position: relative; padding: 10px; overflow-y: auto; }
.chart-wrapper, .chart-instance, .vision-container { width: 100%; height: 100%; }

/* 快照查看 */
.snapshot-viewer { width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; background: #1a1a1a; border-radius: 8px; overflow: hidden; }
.img-box { position: relative; width: 100%; height: 100%; }
.img-box img { width: 100%; height: 100%; object-fit: contain; }
.img-tag { position: absolute; top: 10px; left: 10px; background: #f56c6c; color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.5); }
.no-data { text-align: center; color: #555; display: flex; flex-direction: column; align-items: center; gap: 8px; }

@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>