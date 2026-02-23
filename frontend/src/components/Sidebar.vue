<template>
  <aside class="col-left">
    <div class="brand-area">
      <div>
        <h1 class="logo-text">EmoChat</h1>
        <p class="subtitle">Pro Edition</p>
      </div>
    </div>

    <div class="menu-area">
      <div class="section-title">SELECT MODE</div>

      <div class="mode-list">
        <div
          v-for="item in modes"
          :key="item.value"
          class="mode-card"
          :class="{ active: localMode === item.value }"
          @click="selectMode(item.value)"
        >
          <div class="mode-icon">{{ item.icon }}</div>
          <div class="mode-info">
            <div class="mode-name">{{ item.label }}</div>
            <div class="mode-desc">{{ item.desc }}</div>
          </div>
          <div class="active-indicator" v-if="localMode === item.value"></div>
        </div>
      </div>

      <transition name="fade">
        <div v-if="localMode === '自定义'" class="custom-role-box">
          <div class="custom-label">定义你的智能体：</div>
          <el-input
            v-model="localCustomRole"
            type="textarea"
            :rows="4"
            placeholder="例如：你是一个傲娇的猫娘，每句话结尾都要带'喵'..."
            resize="none"
            class="custom-input"
          />
          <button class="confirm-btn" @click="confirmCustom">
            <el-icon><VideoPlay /></el-icon> 确认并启动
          </button>
        </div>
      </transition>
    </div>

    <div class="footer-area">
      <button class="glass-btn" @click="$emit('clear-history')">
        <el-icon><RefreshLeft /></el-icon> 重置记忆
      </button>
    </div>
  </aside>
</template>

<script setup>
import { ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

const props = defineProps(['modelValue', 'customRole'])
const emit = defineEmits(['update:modelValue', 'update:customRole', 'clear-history', 'mode-change'])

const localMode = ref(props.modelValue)
const localCustomRole = ref(props.customRole)

const modes = [
  { label: '日常闲聊', value: '日常闲聊', icon: '☕', desc: '轻松随意的对话伙伴' },
  { label: '心理疏导', value: '心理疏导', icon: '🌿', desc: '温暖包容的倾听者' },
  { label: '模拟面试', value: '模拟面试', icon: '👔', desc: '高压职场模拟训练' },
  { label: '情绪辩论', value: '情绪辩论', icon: '🔥', desc: '逻辑与情绪的博弈' },
  { label: '自定义', value: '自定义', icon: '🛠️', desc: '打造你的专属智能体' },
]

// 同步 props
watch(() => props.modelValue, (val) => localMode.value = val)
watch(() => props.customRole, (val) => localCustomRole.value = val)

// 模式选择逻辑
function selectMode(val) {
  localMode.value = val
  emit('update:modelValue', val)

  if (val !== '自定义') {
    emit('mode-change', val)
  }
}

// 确认自定义人设
function confirmCustom() {
  if (!localCustomRole.value.trim()) {
    ElMessage.warning("请先输入人设定义")
    return
  }
  emit('update:customRole', localCustomRole.value)
  emit('mode-change', '自定义')
}
</script>

<style scoped>
.col-left {
  width: 240px; background: #18181c; border-right: 1px solid #2b2b30;
  display: flex; flex-direction: column; padding: 24px 16px; flex-shrink: 0;
}

.brand-area { display: flex; align-items: center; gap: 12px; margin-bottom: 30px; padding-left: 8px; }
.logo-icon { font-size: 28px; }
.logo-text { margin: 0; font-size: 20px; font-weight: 800; letter-spacing: -0.5px; color: #fff; }
.subtitle { font-size: 10px; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; }

.menu-area { flex: 1; overflow-y: auto; }
.section-title { font-size: 10px; color: #666; margin-bottom: 12px; font-weight: 700; padding-left: 8px; letter-spacing: 1px; }

.mode-list { display: flex; flex-direction: column; gap: 10px; }

.mode-card {
  display: flex; align-items: center; gap: 14px;
  padding: 14px; border-radius: 12px;
  background: #232328; cursor: pointer;
  transition: all 0.2s ease; border: 1px solid transparent;
  position: relative; overflow: hidden;
}
.mode-card:hover { background: #2a2a30; transform: translateY(-1px); }
.mode-card.active {
  background: linear-gradient(135deg, #303036, #25252a);
  border-color: #4c4c52;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.mode-icon { font-size: 22px; filter: none; }
.mode-info { flex: 1; }
.mode-name { font-size: 15px; font-weight: 600; color: #e0e0e0; margin-bottom: 2px; }
.mode-card.active .mode-name { color: #fff; text-shadow: 0 0 10px rgba(255,255,255,0.3); }
.mode-desc { font-size: 11px; color: #888; }
.mode-card.active .mode-desc { color: #aaa; }
.active-indicator {
  position: absolute; right: 0; top: 50%; transform: translateY(-50%);
  width: 4px; height: 20px; background: #409eff; border-radius: 4px 0 0 4px;
  box-shadow: 0 0 8px rgba(64,158,255, 0.8);
}

/* Custom Box */
.custom-role-box { margin-top: 20px; padding: 12px; background: #232328; border-radius: 12px; border: 1px solid #333; animation: slideDown 0.3s ease; }
.custom-label { font-size: 11px; color: #888; margin-bottom: 8px; font-weight: 600; }

:deep(.custom-input .el-textarea__inner) {
  background: #1a1a1e; border: 1px solid #333; color: #ddd; font-size: 13px; border-radius: 8px; padding: 10px; box-shadow: none;
}
:deep(.custom-input .el-textarea__inner:focus) { border-color: #409eff; }

/* 按钮样式 */
.confirm-btn {
  width: 100%; margin-top: 10px; padding: 10px;
  background: linear-gradient(90deg, #409eff, #3a8ee6);
  border: none; border-radius: 6px;
  color: #fff; font-size: 12px; font-weight: 700; cursor: pointer;
  display: flex; align-items: center; justify-content: center; gap: 6px;
  transition: all 0.2s;
}
.confirm-btn:hover { filter: brightness(1.1); transform: scale(1.02); }
.confirm-btn:active { transform: scale(0.98); }

.footer-area { margin-top: 20px; }
.glass-btn {
  width: 100%; padding: 12px; border: 1px solid #333; background: #232328;
  color: #bbb; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px;
  transition: 0.2s; font-size: 13px; font-weight: 600;
}
.glass-btn:hover { background: #2d2d33; color: #fff; border-color: #555; }

@keyframes slideDown { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
</style>