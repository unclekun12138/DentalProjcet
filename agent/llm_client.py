"""
LLM Client - OpenAI-compatible API (DeepSeek / Qwen / etc.)
Usage:
    from llm_client import LLMClient
    llm = LLMClient(api_key="your-key", base_url="https://api.deepseek.com/v1")
    response = llm.chat("你好")
"""
import os
import json
from typing import List, Dict, Optional


class LLMClient:
    """OpenAI-compatible LLM client."""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client if api_key is available."""
        if not self.api_key:
            print("[LLM] No API key set, running in mock mode")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            print(f"[LLM] Connected to {self.base_url}, model={self.model}")
        except ImportError:
            print("[LLM] openai package not installed, running in mock mode")
    
    def chat(self, messages: List[Dict], tools: List[Dict] = None, temperature: float = 0.7) -> str:
        """
        Chat with LLM.
        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            tools: optional function definitions for tool calling
            temperature: creativity
        Returns:
            assistant response text
        """
        if self.client is None:
            return self._mock_response(messages)
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM Error] {str(e)}"
    
    def _mock_response(self, messages: List[Dict]) -> str:
        """Mock response when no API key."""
        last_msg = messages[-1]["content"] if messages else ""
        return f"[Mock LLM] 收到您的问题：{last_msg[:100]}\n配置 API key 后即可获得真实回答。"


class DentalRAG:
    """Simple RAG for dental medical knowledge."""
    
    def __init__(self):
        # Dental knowledge base: key = searchable keywords (Chinese + English),
        # value = concise knowledge entry (2-4 sentences) for RAG retrieval.
        self.knowledge = {
            # ========== 1. 常见口腔疾病 (Common Oral Diseases) ==========
            "龋齿_caries_蛀牙_dental_caries": (
                "龋齿（Dental Caries，俗称蛀牙）是在细菌、食物、宿主和时间多因素作用下，牙体硬组织发生的慢性进行性破坏疾病。"
                "主要致龋菌为变形链球菌，菌斑生物膜发酵碳水化合物产酸，导致牙釉质和牙本质脱矿。"
                "临床表现为牙面白垩色斑块、龋洞、对冷热酸甜敏感，不及时治疗可进展为牙髓炎、根尖周炎甚至牙体缺损丧失。"
            ),
            "龋齿分级_浅龋_中龋_深龋_caries_classification": (
                "龋齿按病变深度分为浅龋（釉质龋）、中龋（牙本质浅层龋）和深龋（牙本质深层龋）。"
                "浅龋一般无自觉症状，牙面呈白垩色或褐色斑；中龋遇冷热酸甜刺激敏感；深龋有明显食物嵌塞痛和冷热刺激痛，刺激去除后症状立即消失。"
                "深龋接近牙髓时需注意鉴别是否已发生牙髓炎。"
            ),
            "牙髓炎_pulpitis_牙髓感染": (
                "牙髓炎（Pulpitis）多由深龋未及时治疗，细菌感染牙髓所致，分为可复性牙髓炎和不可复性牙髓炎。"
                "典型症状为自发性、阵发性剧痛，夜间痛加重，温度刺激诱发并持续较长时间，疼痛常不能定位。"
                "治疗需行根管治疗（Root Canal Therapy），去除感染牙髓后充填根管。"
            ),
            "牙周炎_periodontitis_牙周病": (
                "牙周炎（Periodontitis）是由牙菌斑生物膜引起的牙周支持组织（牙龈、牙周膜、牙槽骨、牙骨质）的慢性感染性疾病。"
                "主要表现为牙龈红肿出血、牙周袋形成、附着丧失、牙槽骨吸收，晚期可出现牙齿松动、移位甚至脱落。"
                "是中国成年人失牙的首要原因之一，治疗以牙周基础治疗（洁治、刮治）为核心。"
            ),
            "牙龈炎_gingivitis_牙龈出血": (
                "牙龈炎（Gingivitis）是局限于牙龈组织的炎症，病变未累及深层牙周组织，无附着丧失和牙槽骨吸收。"
                "主要表现为牙龈红肿、刷牙或咬硬物时出血，牙龈缘变厚、龈乳头圆钝。"
                "牙龈炎为可逆性病变，通过洁治（Scaling）彻底清除菌斑牙石后可完全恢复正常。"
            ),
            "智齿_wisdom_tooth_冠周炎_pericoronitis": (
                "智齿冠周炎（Pericoronitis）是智齿（第三磨牙）萌出不全或阻生时，牙冠周围软组织发生的炎症。"
                "以下颌智齿最为常见，表现为磨牙后区胀痛、进食吞咽加重，可伴张口受限、面颊肿胀、发热等全身症状。"
                "急性期以局部冲洗上药、抗感染为主，炎症消退后应尽早拔除阻生智齿。"
            ),
            "阻生智齿_impacted_wisdom_tooth_阻生齿": (
                "阻生智齿（Impacted Wisdom Tooth）是指由于邻牙、骨或软组织阻碍，智齿只能部分萌出或完全不能萌出。"
                "根据阻生方向可分为近中阻生、远中阻生、水平阻生、垂直阻生和倒置阻生等。"
                "阻生智齿可导致冠周炎反复发作、邻牙龋坏、牙列拥挤和囊肿形成，通常建议预防性拔除。"
            ),
            "牙列不齐_malocclusion_错颌畸形_牙列拥挤": (
                "牙列不齐（Malocclusion，错颌畸形）是指儿童生长发育过程中，由先天遗传或后天环境因素导致的牙齿、颌骨、颅面畸形。"
                "表现为牙列拥挤、牙间隙、前牙反颌、深覆合、深覆盖、开颌等，不仅影响美观，还可影响咀嚼功能、发音和牙周健康。"
                "矫治时机一般为恒牙列初期（11-14岁），但成人也可进行正畸治疗。"
            ),
            "颞下颌关节紊乱_tmd_tmj_颞下颌关节": (
                "颞下颌关节紊乱病（Temporomandibular Disorders, TMD）是累及颞下颌关节和/或咀嚼肌的一组疾病的总称。"
                "主要症状包括关节弹响、张口受限、关节区及咀嚼肌疼痛、张口偏斜，常与精神压力、夜磨牙、咬合异常、外伤有关。"
                "多数为自限性，以保守治疗为主（热敷、理疗、咬合板、药物），严重者需手术。"
            ),
            "口腔溃疡_recurrent_aphthous_ulcer_复发性口疮": (
                "复发性阿弗他溃疡（Recurrent Aphthous Ulcer）是最常见的口腔黏膜病，表现为反复发作的圆形或椭圆形浅表溃疡，灼痛明显。"
                "病因不明，与免疫因素、遗传、疲劳、压力、维生素缺乏和局部创伤有关，具有自限性，一般7-14天自愈。"
                "治疗以消炎止痛、促进愈合为主，超过两周不愈的溃疡需活检排除恶变。"
            ),

            # ========== 2. 治疗方式 (Treatment Methods) ==========
            "补牙_filling_充填_restoration_树脂充填": (
                "补牙（充填治疗，Filling/Restoration）是通过去除龋坏组织、制备洞型后，用牙科材料充填恢复牙齿形态和功能的方法。"
                "常用材料包括复合树脂（Composite Resin，美观、粘结性好）、玻璃离子（释放氟、防龋）和银汞合金（强度高、不美观）。"
                "适用于龋齿、牙体外伤缺损等未累及牙髓的病变，深龋需垫底护髓后再充填。"
            ),
            "根管治疗_root_canal_therapy_rct_抽神经": (
                "根管治疗（Root Canal Therapy, RCT）是治疗牙髓病和根尖周病的首选方法，通过清除根管内感染的牙髓组织，预备、消毒后充填根管。"
                "一般需2-3次就诊，步骤包括开髓、根管预备、根管消毒和根管充填（牙胶尖+封闭剂）。"
                "根管治疗后的牙齿因失去神经营养供应而变脆，建议全冠修复以防止劈裂。"
            ),
            "洗牙_洁治_scaling_牙周洁治": (
                "洗牙（牙周洁治，Scaling）是用超声波或手用器械去除牙龈上牙石、菌斑和色素的口腔保健和治疗手段。"
                "建议健康人群每6-12个月洁治一次，牙周炎患者需更频繁并配合龈下刮治（SRP）。"
                "洗牙后短期内牙面敏感、牙缝增大属正常现象，不会损伤牙齿，是预防和治疗牙周炎的基础措施。"
            ),
            "拔牙_extraction_拔除_ tooth_extraction": (
                "拔牙（Tooth Extraction）是去除无法保留的患牙的外科操作，适应症包括严重龋坏无法修复、晚期牙周病松动牙、阻生智齿、正畸减数牙等。"
                "拔牙后需咬紧棉球30分钟止血，24小时内不漱口不吸吮伤口，避免进食过热食物，预防干槽症。"
                "禁忌证包括急性感染期未控制、严重血液病、未控制的高血压糖尿病、放疗后3-5年内等。"
            ),
            "种植牙_dental_implant_种植体_人工牙根": (
                "种植牙（Dental Implant）是将纯钛种植体（Titanium Implant）植入牙槽骨内作为人工牙根，待骨结合（Osseointegration）后上部修复牙冠的修复方式。"
                "标准流程为：一期手术植入种植体→等待3-6个月骨结合→二期安装基台→取模修复牙冠。"
                "种植牙不损伤邻牙、咀嚼效率高、舒适度好，被誉为人类的第三副牙齿，但对骨量和全身健康条件要求较高。"
            ),
            "正畸_托槽_braces_fixed_orthodontics_固定矫正": (
                "固定正畸（Fixed Orthodontics/Braces）是通过粘结在牙面的托槽（Bracket）和弓丝（Archwire）施加温和矫治力，使牙齿移动到理想位置的方法。"
                "托槽材料包括金属托槽、陶瓷托槽（美观）和自锁托槽（疗程短、舒适）。"
                "疗程一般1.5-3年，需每月复诊加力，矫治结束后需佩戴保持器（Retainer）防止复发。"
            ),
            "正畸_隐形矫正_clear_aligner_invisalign_隐形牙套": (
                "隐形矫正（Clear Aligners，如隐适美Invisalign）是通过一系列透明医用高分子材料矫治器，逐步移动牙齿的矫治方式。"
                "美观、可自行摘戴、口腔卫生维护方便，适用于轻中度牙列拥挤、牙间隙和轻中度咬合问题。"
                "要求患者每日佩戴20-22小时，每1-2周更换一副矫治器，重度错颌畸形仍需固定矫治。"
            ),
            "修复_全冠_crown_牙冠_牙套": (
                "全冠（Crown）是覆盖整个牙冠表面的修复体，用于恢复大面积牙体缺损、根管治疗后牙的保护、美观改善和固定桥固位体。"
                "材料包括全瓷冠（Esthetic，美观通透，前牙首选）、金属烤瓷冠（PFM，强度高）和全金属冠（后牙咬合力量大者）。"
                "牙体需预备磨除约1-1.5mm间隙，取模后制作，试戴粘接完成。"
            ),
            "修复_固定桥_fixed_bridge_固定桥": (
                "固定桥（Fixed Bridge）是利用缺牙两侧的天然牙（基牙，Abutment）作为支持，通过桥体（Pontic）恢复缺失牙的固定修复方式。"
                "适用于少数牙缺失（通常1-3颗）、基牙条件良好的病例，患者不能自行摘戴。"
                "缺点是需磨除两侧健康基牙的牙体组织，缺牙区牙槽骨缺乏功能刺激易发生吸收。"
            ),
            "修复_活动义齿_removable_denture_活动假牙": (
                "活动义齿（Removable Denture）是患者可自行摘戴的修复体，分为局部义齿（RPD）和全口义齿（Complete Denture）。"
                "局部义齿利用卡环和基托固位，磨除牙体少、适用范围广、费用低；全口义齿依靠基托吸附力和大气负压固位。"
                "缺点是舒适度和咀嚼效率较低，需适应期，长期使用可能加速牙槽骨吸收。"
            ),

            # ========== 3. FDI 编号系统 ==========
            "fdi_fdi编号_牙位编号_两位数_notation": (
                "FDI牙位记录系统（FDI Notation，国际牙科联合会系统）采用两位阿拉伯数字记录恒牙和乳牙牙位。"
                "第一位数字代表象限（Quadrant），第二位数字代表牙位从中切牙向后的序列编号。"
                "该系统为国际通用标准，已被WHO采用，便于病历书写、计算机录入和影像定位。"
            ),
            "fdi恒牙象限_象限_quadrant_1_2_3_4": (
                "恒牙FDI四象限：1象限=右上（右上颌右侧），2象限=左上（左上颌左侧），3象限=左下（左下颌左侧），4象限=右下（右下颌右侧）。"
                "记录顺序从患者的右侧前牙区开始，沿上颌牙弓向左，再沿下颌牙弓向右，形成顺时针方向。"
                "牙医从对面观察患者时，患者的右上对应牙医的左侧。"
            ),
            "fdi牙位编号_恒牙_1到8_中切牙_磨牙": (
                "恒牙第二位数字1-8代表从中切牙到第三磨牙的顺序：1=中切牙，2=侧切牙，3=尖牙（犬齿），4=第一前磨牙，5=第二前磨牙，6=第一磨牙，7=第二磨牙，8=第三磨牙（智齿）。"
                "例如11=右上中切牙，16=右上第一磨牙，26=左上第一磨牙，36=左下第一磨牙，46=右下第一磨牙。"
                "记录时不加标点，直接写两位数如11、36、48。"
            ),
            "fdi乳牙编号_乳牙_象限_5_6_7_8": (
                "乳牙FDI编号系统象限编号为5、6、7、8：5=右上乳牙象限，6=左上乳牙象限，7=左下乳牙象限，8=右下乳牙象限。"
                "乳牙第二位数字1-5代表从中切牙到第二乳磨牙：1=乳中切牙，2=乳侧切牙，3=乳尖牙，4=第一乳磨牙，5=第二乳磨牙。"
                "例如51=右上乳中切牙，85=右下第二乳磨牙，用于儿童乳牙列的牙位记录。"
            ),
            "fdi举例_常见牙位_11_16_26_36_46": (
                "常见恒牙FDI编号示例：11=右上中切牙，12=右上侧切牙，13=右上尖牙，14/15=右上第一/二前磨牙，16=右上第一磨牙（六龄齿），17/18=右上第二/三磨牙。"
                "对称牙位：16与26（左右上六龄齿）、36与46（左右下六龄齿）为口腔内最重要的咀嚼牙，萌出最早、患龋率最高。"
                "38=左下智齿，48=右下智齿，18/28=上颌智齿，是临床最常拍摄CBCT定位的牙位。"
            ),

            # ========== 4. CBCT 适应症 ==========
            "cbct_概述_原理_cone_beam_ct_锥形束ct": (
                "CBCT（Cone Beam Computed Tomography，锥形束CT）是口腔医学专用的三维影像设备，利用锥形束X线和面积探测器获得三维容积重建图像。"
                "相比传统螺旋CT，CBCT辐射剂量更低（约为曲面断层的数倍至十余倍，远低于医用CT）、空间分辨率更高、成像时间短。"
                "可获得矢状位、冠状位、轴位任意断面及三维重建图像，是口腔颌面外科、种植、正畸、根管的重要诊断工具。"
            ),
            "cbct种植术前_种植评估_骨高度_骨宽度_骨量": (
                "CBCT是种植牙术前评估的金标准影像：可精确测量缺牙区牙槽嵴顶到下颌神经管（Inferior Alveolar Nerve）、上颌窦底（Maxillary Sinus）的骨高度和颊舌向骨宽度。"
                "用于评估骨量是否足够、是否需要植骨（Grafting）或上颌窦提升（Sinus Lift），并设计种植体三维位置和角度。"
                "可配合数字化软件制作种植导板（Surgical Guide），实现微创精准种植。"
            ),
            "cbct阻生齿_阻生智齿定位_牙根与神经管关系": (
                "CBCT用于阻生智齿特别是下颌低位阻生、水平阻生的定位：可清晰显示智齿牙根形态、分叉情况以及与下牙槽神经管的三维位置关系。"
                "术前评估神经管与牙根是否紧贴、横跨或位于牙根之间，可显著降低术中神经损伤（下唇麻木）风险。"
                "也用于多生牙（Supernumerary Tooth）、埋伏尖牙的定位和入路设计。"
            ),
            "cbct颞下颌关节_tmj关节髁突_cbct": (
                "CBCT是颞下颌关节（TMJ）硬组织检查的重要手段：可显示髁突（Condyle）的形态、骨质改变，包括骨质增生、吸收、破坏、骨赘形成和关节间隙变化。"
                "常用于TDD患者、关节强直、关节外伤骨折和肿瘤的评估。"
                "软组织（关节盘）病变需结合MRI检查，CBCT不适合评估关节盘位置。"
            ),
            "cbct根管治疗_疑难根管_根管侧穿_根尖周病变": (
                "CBCT辅助疑难根管治疗：可发现根管侧穿、根管内器械分离（Separated Instrument）、额外根管（如MB2）、根尖周骨质破坏范围和根管解剖变异。"
                "用于常规X线片无法明确的根管钙化、根管台阶、根尖吸收和根折诊断。"
                "可辅助手术根管治疗（Apicoectomy）的切口和去骨范围设计。"
            ),
            "cbct颌骨病变_囊肿_肿瘤_颌骨囊性病变": (
                "CBCT用于颌骨病变的诊断和范围评估：包括颌骨囊肿（Radicular Cyst、Odontogenic Keratocyst）、成釉细胞瘤（Ameloblastoma）、颌骨骨髓炎和肿瘤的三维定位。"
                "可显示病变与牙根、神经管、上颌窦、鼻腔的关系，辅助手术方案设计。"
                "CBCT对软组织分辨率有限，怀疑恶性肿瘤时仍需结合螺旋CT和MRI。"
            ),

            # ========== 5. 口扫（口腔扫描）适应症 ==========
            "口扫_ios_intraoral_scanner_口内扫描_数字化印模": (
                "口扫（Intraoral Scanner, IOS，口内扫描仪）是通过光学探头直接在口腔内获取牙列及软组织三维数字模型的设备，替代传统硅橡胶印模。"
                "优点包括无取模不适感、即时可视化、数字模型便于存储传输、可与CAD/CAM系统直接对接制作修复体。"
                "数据以STL/PLY格式输出，广泛应用于正畸、修复、种植和数字咬合分析。"
            ),
            "口扫正畸_正畸方案设计_隐形牙套_数字模型": (
                "口扫是隐形正畸（Clear Aligner）方案设计的标准数据来源：口扫获取的数字牙列模型导入矫治软件，进行牙齿移动模拟和矫治器设计。"
                "也用于固定正畸的术前术后模型分析、拥挤度测量、Bolton指数分析和排牙试验。"
                "相比传统石膏模型，数字模型可直接测量、叠加对比和远程会诊。"
            ),
            "口扫修复体设计_cad_cam_全冠_修复体": (
                "口扫获取的预备后牙体数字模型，结合对颌牙列和咬合记录数据，导入CAD软件设计修复体（全冠、嵌体、桥），再由CAM切削或3D打印制作。"
                "实现椅旁即刻修复（如CEREC系统），一次就诊完成补牙/嵌体修复。"
                "精度满足单冠和短桥要求，长桥和全口修复需注意扫描误差累积。"
            ),
            "口扫种植导板_种植导板设计_surgical_guide": (
                "口扫获取缺牙区软组织和对颌数字模型，与CBCT骨组织数据配准（Digital Workflow），在软件中规划种植体位置并设计种植手术导板。"
                "导板通过3D打印制作，术中引导钻孔方向和深度，实现不翻瓣或少翻瓣的微创种植。"
                "数字化种植流程提高了种植位置精度，减少术后反应。"
            ),
            "口扫咬合记录_数字咬合_颌位关系_bite": (
                "口扫可采集最大牙尖交错位（ICP）的咬合记录，并结合面部扫描或下颌运动轨迹，进行数字化咬合分析。"
                "用于评估咬合接触点分布、早接触、咬合干扰，辅助修复和正畸后的咬合调整。"
                "全口咬合重建病例需配合面弓（Facebow）转移和颌架（Articulator）分析。"
            ),

            # ========== 6. 种植适应症/禁忌症 ==========
            "种植适应症_单颗牙缺失_single_tooth_missing": (
                "单颗牙缺失是种植牙的最佳适应症：缺失区邻牙健康时，种植牙不需要磨除邻牙牙体组织，避免了固定桥对邻牙的损伤。"
                "种植修复后可恢复咀嚼功能至天然牙的90%以上，外形美观自然。"
                "适用于18岁以上颌骨发育完成、缺牙区骨量和咬合关系基本正常的患者。"
            ),
            "种植适应症_多颗牙缺失_multiple_teeth_missing": (
                "多颗牙缺失可采用种植固定桥或多颗独立种植体支持的修复方式，避免传统活动义齿的大基托和卡环。"
                "对于游离端缺失（Kennedy I类，后牙末端缺失），传统修复无法有效承担咀嚼力，种植修复具有显著优势。"
                "种植体数量和位置由缺牙间隙、咬合力量和骨条件共同决定。"
            ),
            "种植适应症_全口种植_all_on_4_全口无牙颌": (
                "全口无牙颌（Edentulism）可采用All-on-4/Technique：利用前牙区4颗种植体（远中种植体倾斜30-45°）即刻负荷固定全口修复桥。"
                "优点是减少植骨手术、缩短疗程、术后当天即可戴固定临时牙，显著改善老年患者生活质量。"
                "对颌骨严重萎缩患者，可能需要Onlay植骨、上颌窦提升或短种植体/窄种植体方案。"
            ),
            "种植禁忌症_骨量不足_insufficient_bone_植骨": (
                "骨量不足是种植最常见的相对禁忌症：缺牙区牙槽嵴宽度<6mm或高度不足时，种植体无法获得足够初期稳定性和骨结合。"
                "可通过GBR（引导骨再生，Guided Bone Regeneration）、上颌窦提升术（Sinus Lift）、onlay植骨、牵张成骨等方法增量后再种植。"
                "严重全身因素未控制时应先治疗原发病再评估种植。"
            ),
            "种植禁忌症_糖尿病_diabetes_血糖控制": (
                "未控制的糖尿病（HbA1c>7%-8%）是种植相对禁忌症：高血糖影响创口愈合、增加感染风险、降低种植体骨结合率。"
                "糖尿病患者血糖控制稳定后（HbA1c<7%）可在严密监控下进行种植，术中术后需预防性使用抗生素并缩短随访间隔。"
                "术后需特别注意口腔卫生维护和牙周状况监控。"
            ),
            "种植禁忌症_吸烟_smoking_烟草": (
                "吸烟是种植失败的重要危险因素：尼古丁和一氧化碳影响微循环和成骨细胞活性，吸烟患者种植体失败率显著高于非吸烟者。"
                "一般建议术前戒烟至少2周、术后戒烟至骨结合完成（约3-6个月），每日吸烟量>10支者失败风险明显升高。"
                "吸烟患者应缩短随访周期，加强牙周维护。"
            ),
            "种植禁忌症_全身因素_放疗_双膦酸盐_抗凝药": (
                "种植绝对/相对禁忌症还包括：近期心肌梗死或脑卒中（6个月内）、未控制的严重高血压/糖尿病、化疗期恶性肿瘤患者。"
                "头颈部放疗后3-5年内颌骨坏死（ORN）风险高，应谨慎种植；静脉双膦酸盐（Bisphosphonate）治疗患者存在颌骨骨坏死（MRONJ）风险，需停药评估。"
                "长期双抗凝药患者需与内科协调围手术期用药。"
            ),

            # ========== 7. 正畸适应症 ==========
            "正畸适应症_牙列拥挤_crowding_拥挤": (
                "牙列拥挤（Crowding）是最常见的错颌畸形病因：牙量骨量不调，牙弓长度不足以容纳全部牙齿，导致牙齿重叠、错位、扭转。"
                "轻度拥挤（<5mm）可通过邻面去釉（IPR）解决；中度拥挤（5-8mm）需拔牙或扩弓；重度拥挤（>8mm）通常需要拔牙矫治。"
                "拥挤不易清洁，是龋齿和牙周病的好发因素。"
            ),
            "正畸适应症_牙间隙_spacing_diastema_牙缝": (
                "牙间隙（Spacing/Diastema）表现为牙列中存在散在间隙或正中分离（Midline Diastema，上颌中切牙间隙）。"
                "病因包括牙量小、颌骨发育过大、唇系带附着过低、不良习惯等。"
                "矫治通过关闭间隙集中缝隙后保持，前牙区过大间隙矫治后需长期保持器维持，必要时配合贴面修复。"
            ),
            "正畸适应症_反颌_地包天_underbite_crossbite": (
                "前牙反颌（Anterior Crossbite，俗称地包天/兜齿）表现为下前牙位于上前牙外侧，是我国常见的骨性错颌畸形。"
                "功能性反颌早期（3-5岁）矫治效果好，骨性反颌需在生长发育期（女孩10-12岁、男孩12-14岁）前方牵引，成年后严重骨性者需正畸-正颌联合治疗。"
                "反颌影响颌骨发育和咀嚼功能，应尽早筛查干预。"
            ),
            "正畸适应症_深覆合_deep_overbite_深覆盖": (
                "深覆合（Deep Overbite）指上前牙覆盖下前牙超过冠长1/3，严重时下前牙咬及上腭黏膜；深覆盖（Overjet）指上前牙前突水平距离>3mm（俗称龅牙）。"
                "深覆合可导致咬合创伤、牙龈退缩、关节症状；深覆盖影响美观和发音。"
                "矫治通过压低下前牙、升高后牙或内收前牙纠正，成人骨性畸形需正颌手术。"
            ),
            "正畸适应症_开颌_open_bite_开合": (
                "开颌（Open Bite）表现为上下颌牙齿在垂直方向不能咬合接触，前牙开颌最常见，多与口呼吸、吮指、吐舌习惯有关。"
                "影响切割食物和发音，严重者后牙也无接触。"
                "儿童早期破除不良习惯后可自行改善；成人骨性开颌治疗难度大，常需正畸-正颌联合。"
            ),

            # ========== 8. 修复适应症 ==========
            "修复适应症_牙体缺损_tooth_defect_缺损": (
                "牙体缺损（Tooth Defect）指牙硬组织不同程度的外形和结构破坏，常见病因是龋齿、外伤、磨损、楔状缺损和发育畸形。"
                "修复方案选择：缺损小→充填（树脂）；缺损大累及牙尖→嵌体/高嵌体（Inlay/Onlay）；根管治疗后或大面积缺损→全冠（Crown）。"
                "设计时需兼顾剩余牙体抗力形、固位形和咬合保护。"
            ),
            "修复适应症_牙列缺损_tooth_absence_partial_缺损": (
                "牙列缺损（Partial Edentulism）指牙列内有部分牙缺失，可采用种植修复、固定桥或活动局部义齿（RPD）三种方案。"
                "选择依据包括：缺失牙数目和位置、邻牙健康状况、咬合关系、患者舒适度要求和经济条件。"
                "种植修复不损伤邻牙、咀嚼效率最高，是目前多数牙列缺损患者的首选方案。"
            ),
            "修复适应症_牙列缺失_complete_edentulism_全口无牙": (
                "牙列缺失（Complete Edentulism）指上下颌牙列全部缺失，传统修复为全口总义齿（Complete Denture），依靠基托吸附力固位。"
                "牙槽嵴严重吸收的患者传统义齿固位差、咀嚼效率低，可采用种植覆盖义齿（Locator/球帽附着体）或种植固定全口桥（All-on-4）。"
                "全口义齿修复需在拔牙后2-3个月牙槽嵴稳定后进行，定期重衬（Relining）维护。"
            ),
        }

    def retrieve(self, query: str) -> str:
        """Retrieve relevant knowledge entries by matching query against
        keyword keys (split by '_') and against the knowledge content itself."""
        if not query or not query.strip():
            return "未找到相关知识，请咨询专业医生。"
        query_lower = query.lower().strip()
        results = []
        seen = set()
        for key, value in self.knowledge.items():
            # Match 1: any underscore-separated keyword in key is a substring of query
            key_words = [w for w in key.split("_") if w]
            key_match = any(
                word.lower() in query_lower for word in key_words
            )
            # Match 2: query (or a significant part) appears in the value content
            value_match = query_lower in value.lower()
            if key_match or value_match:
                if value not in seen:
                    seen.add(value)
                    results.append(value)
        if not results:
            return "未找到相关知识，请咨询专业医生。"
        return "\n---\n".join(results)


# System prompt for dental analysis agent
SYSTEM_PROMPT = """你是一个专业的口腔影像分析助手。你的职责是：
1. 分析用户上传的口腔影像（CBCT或口扫点云）
2. 识别牙齿数量、位置和状态
3. 生成结构化的分析报告
4. 回答用户关于口腔健康的问题

注意：
- 你的分析仅供参考，不能替代医生诊断
- 如果不确定，建议用户咨询专业牙医
- 使用专业但易懂的语言
"""


if __name__ == "__main__":
    # Test
    llm = LLMClient()
    print(llm.chat([{"role": "user", "content": "什么是FDI编号系统？"}]))
    
    rag = DentalRAG()
    print(f"\n知识库总条目数: {len(rag.knowledge)}")
    print("\n=== RAG test: 智齿 ===")
    print(rag.retrieve("智齿"))
    print("\n=== RAG test: FDI ===")
    print(rag.retrieve("FDI"))
    print("\n=== RAG test: 种植牙 ===")
    print(rag.retrieve("种植牙"))
