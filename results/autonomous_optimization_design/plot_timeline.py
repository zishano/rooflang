from pathlib import Path
import json
import argparse
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'rooflang-moe-scheduling-mvp'
import matplotlib.pyplot as plt
from matplotlib import font_manager
parser = argparse.ArgumentParser(description='Plot the scheduling validation timeline.')
parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent / 'validation')
parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
args = parser.parse_args()
root = args.results_dir
r=json.loads((root/'results.json').read_text())
if args.font.is_file():
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(args.font)).get_name()
colors={'routed_combine':'#2563eb','shared_down':'#ea8a23','moe_add':'#159b80'}
labels={'routed_combine':'专家结果合并（含远程读）','shared_down':'共享专家 down GEMM','moe_add':'两分支相加'}
fig,axes=plt.subplots(2,1,figsize=(12,5.8),sharex=True)
for ax,row,title in zip(axes,r['rows'][:2],['原计划：同一 stream 串行','候选：独立 shared_down 使用 stream 1']):
 for e in row['trace']:
  ax.barh(e['stream'],e['end_us']-e['start_us'],left=e['start_us'],height=.48,color=colors[e['name']])
 ax.set_yticks([0,1],['stream 0','stream 1']);ax.set_ylim(1.5,-.6)
 ax.set_title(f'{title}    完成：{row["total_us"]:.3f} μs',loc='left',fontsize=12)
 ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 ax.spines[['top','right']].set_visible(False)
axes[-1].set_xlabel('模拟时间（μs），所有根输入在 t=0 已就绪')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=colors[n],label=labels[n]) for n in colors],loc='upper center',ncol=3,bbox_to_anchor=(.5,.93),frameon=False)
fig.suptitle('DSV4 Flash MoE 片段：只改变 stream，观察重叠与代价',fontsize=15,y=.99)
fig.subplots_adjust(top=.78,bottom=.17,hspace=.45,left=.1,right=.97)
fig.text(.1,.04,'片段延迟降低 26.63%；峰值 HBM 增加 128 KiB；FLOPs / 通信字节不变。\n这是隔离片段的仿真结果，尚未验证整模型、数值输出和实机收益。',fontsize=10,color='#475569')
fig.savefig(root/'timeline.png',dpi=180)
svg_path = root / 'timeline.svg'
fig.savefig(svg_path, metadata={'Date': None})
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
