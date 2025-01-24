from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
import logging
import re
import json
from collections import Counter
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)

class EnhancedPlanning(PlanningBase):
    """Enhanced planning with dynamic platform adaptation"""
    
    def __init__(self, llm):
        super().__init__(llm=llm)
    
    def __call__(self, task_description):
        platform = task_description.get('source', 'yelp')
        self.plan = [
            {
                'description': f'Fetch {platform} user profile',
                'tool_use': {'user_id': task_description['user_id']}
            },
            {
                'description': f'Retrieve {platform} item details',
                'tool_use': {'item_id': task_description['item_id']}
            },
            {
                'description': 'Analyze historical patterns',
                'tool_use': {'review_analysis': True}
            }
        ]
        return self.plan

class EnhancedReasoning(ReasoningBase):
    """Enhanced reasoning with dynamic prompt construction"""
    
    def __init__(self, llm):
        super().__init__(profile_type_prompt='', memory=None, llm=llm)
        
    def _get_platform_prompt(self, source):
        platform_config = {
            "yelp": {
                "rating_criteria": ["服务质量", "环境卫生", "性价比", "地理位置"],
                "focus_aspects": ["菜品特色", "服务态度", "用餐体验"],
                "style_tips": "口语化表达，使用生动形象的描述"
            },
            "amazon": {
                "rating_criteria": ["产品质量", "功能表现", "性价比", "包装体验"],
                "focus_aspects": ["产品功能", "使用体验", "耐用性"],
                "style_tips": "注重技术细节，保持客观专业"
            },
            "goodreads": {
                "rating_criteria": ["故事情节", "人物塑造", "文笔风格", "思想深度"],
                "focus_aspects": ["情节发展", "角色刻画", "主题表达"],
                "style_tips": "体现文学分析，突出情感共鸣"
            }
        }
        return platform_config.get(source.lower(), platform_config["yelp"])

    def _extract_keywords(self, reviews, top_n=3):
        texts = [r.get('text', '') for r in reviews[:10]]
        words = [word.lower() for text in texts for word in word_tokenize(text) if len(word) > 2]
        return [w[0] for w in Counter(words).most_common(top_n)]

    def _build_prompt(self, task_context):
        platform = task_context['item_data'].get('source', 'yelp').lower()
        config = self._get_platform_prompt(platform)
        
        # 用户特征分析
        user_features = [
            f"历史评分：{task_context['user_pattern']['average_rating']:.1f}星" if task_context['user_pattern']['average_rating'] else "",
            f"评论数量：{task_context['user_data'].get('review_count', 0)}",
            f"常用词汇：{', '.join(task_context['user_pattern']['frequent_words'][:3])}"
        ]
        
        # 商品特征提取
        item = task_context['item_data']
        item_features = []
        if platform == "yelp":
            item_features = [
                f"类别：{item.get('categories', '')}",
                f"属性：{', '.join(list(json.loads(item.get('attributes', '{}')).keys())[:3])}"
            ]
        elif platform == "amazon":
            item_features = [
                f"功能：{', '.join(item.get('features', [])[:3])}",
                f"详情：{item.get('description', [''])[0][:100]}..."
            ]
        elif platform == "goodreads":
            item_features = [
                f"描述：{item.get('description', '')[:100]}...",
                f"热门标签：{', '.join([s['name'] for s in item.get('popular_shelves', [])[:3]])}"
            ]
        
        # 构建动态Prompt
        prompt_template = f"""
        **{platform.capitalize()}评论生成任务**
        
        用户特征：
        {', '.join([f for f in user_features if f])}
        
        商品信息：
        {chr(10).join(item_features)}
        
        评分标准（1-5星）：
        {', '.join(config['rating_criteria'])}
        
        写作要求：
        - 重点突出：{', '.join(config['focus_aspects'])}
        - 风格建议：{config['style_tips']}
        - 使用真实用户常用的自然表达
        - 包含至少一个具体的使用场景描述
        - 评分需符合历史模式（平均{task_context['user_pattern']['average_rating']:.1f}星）
        
        参考词汇：
        {', '.join(self._extract_keywords(task_context['reviews_item']))}
        
        输出格式：
        stars: [1.0-5.0]
        review: [50-100字评论，包含具体细节]
        """
        return prompt_template.strip()

    def __call__(self, task_context):
        prompt = self._build_prompt(task_context)
        messages = [{
            "role": "system",
            "content": "你是一个专业的用户行为模拟器，需要生成符合用户历史模式和平台特征的评论。"
        }, {
            "role": "user",
            "content": prompt
        }]
        
        return self.llm(
            messages=messages,
            temperature=0.3,
            max_tokens=512
        )

class EnhancedUserModelingAgent(SimulationAgent):
    """Optimized simulation agent with platform adaptation"""
    
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = EnhancedPlanning(llm=self.llm)
        self.reasoning = EnhancedReasoning(llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        
    def _analyze_user_pattern(self, user_id):
        reviews = self.interaction_tool.get_reviews(user_id=user_id)
        if not reviews:
            return {
                'average_rating': None,
                'frequent_words': []
            }
        
        # 计算评分模式
        ratings = [r.get('stars', 3) for r in reviews]
        avg_rating = sum(ratings)/len(ratings)
        
        # 提取高频词汇
        texts = ' '.join([r.get('text', '') for r in reviews])
        words = [word.lower() for word in word_tokenize(texts) if word.isalpha()]
        freq_words = [w[0] for w in Counter(words).most_common(5)]
        
        return {
            'average_rating': avg_rating,
            'rating_distribution': dict(Counter(ratings)),
            'frequent_words': freq_words
        }
    
    def _parse_output(self, result, user_pattern):
        # 改进的解析方法
        star_match = re.search(r"stars:\s*([1-5](?:\.0)?)", result)
        review_match = re.search(r"review:\s*(.*?)(?=\nstars:|\Z)", result, re.DOTALL)
        
        stars = float(star_match.group(1)) if star_match else (
            user_pattern['average_rating'] if user_pattern['average_rating'] else 3.0
        )
        
        review = review_match.group(1).strip() if review_match else ""
        if not review:
            review = " ".join(user_pattern['frequent_words'][:3]) + "。不错的体验。"
        
        # 确保评分格式正确
        stars = max(1.0, min(5.0, float(stars)))
        return stars, review[:512]

    def workflow(self):
        try:
            # 获取基础数据
            user_data = self.interaction_tool.get_user(self.task['user_id']) or {}
            item_data = self.interaction_tool.get_item(self.task['item_id']) or {}
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            
            # 分析用户模式
            user_pattern = self._analyze_user_pattern(self.task['user_id'])
            
            # 构建任务上下文
            task_context = {
                'user_data': user_data,
                'item_data': item_data,
                'reviews_item': reviews_item,
                'user_pattern': user_pattern
            }
            
            # 生成评论
            result = self.reasoning(task_context)
            
            # 解析结果
            stars, review = self._parse_output(result, user_pattern)
            
            # 后处理：确保评分一致性
            if user_pattern.get('average_rating'):
                rating_diff = abs(stars - user_pattern['average_rating'])
                if rating_diff > 1.5:  # 如果偏离历史平均超过1.5星
                    stars = user_pattern['average_rating']
            
            return {
                "stars": round(stars * 2) / 2,  # 四舍五入到最近的0.5
                "review": review
            }
            
        except Exception as e:
            logging.error(f"Workflow error: {str(e)}")
            return {
                "stars": 3.0,
                "review": "体验符合预期。"
            }

# 使用示例
if __name__ == "__main__":
    simulator = Simulator(data_dir="path/to/data", device="gpu", cache=True)
    
    # 配置多平台测试
    for platform in ["yelp", "amazon", "goodreads"]:
        logging.info(f"Testing platform: {platform}")
        simulator.set_task_set(platform)
        simulator.set_agent(OptimizedAgent)
        simulator.set_llm(InfinigenceLLM(api_key="your_api_key"))
        
        # 运行测试
        outputs = simulator.run_simulation(number_of_tasks=50)
        evaluation = simulator.evaluate()
        
        print(f"{platform} 评估结果:")
        print(f"评分MAE: {evaluation['star_mae']:.3f}")
        print(f"评论质量: {evaluation['review_quality']:.3f}")