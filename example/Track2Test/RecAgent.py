import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import re
import logging
import time
logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

Task: {task_description}
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}
'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        return reasoning_result

class OptimizedRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)
        
    def get_platform(self):
        # 获取平台类型（根据第一个候选商品判断）
        first_item = self.interaction_tool.get_item(self.task['candidate_list'][0])
        platform = first_item.get('source', 'yelp').lower()
        return platform
    
    def _process_item(self, item, platform):
        """平台特定的商品信息处理"""
        base_info = {
            'item_id': item['item_id'],
            'name': item.get('name') or item.get('title'),
            'rating': item.get('stars') or item.get('average_rating')
        }
        
        if platform == 'yelp':
            return {
                **base_info,
                'categories': item.get('categories', '').split(', ')[:3],
                'features': self._extract_yelp_features(item)
            }
        elif platform == 'amazon':
            return {
                **base_info,
                'brand': item.get('details', {}).get('Brand'),
                'price': item.get('price'),
                'features': item.get('features', [])[:5]
            }
        elif platform == 'goodreads':
            return {
                **base_info,
                'author': item.get('authors', [{}])[0].get('author_id'),
                'popular_shelves': [shelf['name'] for shelf in item.get('popular_shelves', [])[:3]]
            }
        else:
            return base_info

    def _extract_yelp_features(self, item):
        """提取Yelp商家关键特征"""
        features = []
        attrs = item.get('attributes', {})
        if 'BusinessParking' in attrs:
            parking = eval(attrs['BusinessParking'].replace("u'", "'"))
            features.append('Parking: ' + ', '.join([k for k,v in parking.items() if v]))
        if 'RestaurantsPriceRange2' in attrs:
            features.append(f'Price Range: {attrs["RestaurantsPriceRange2"]}/4')
        return features

    def _get_user_history(self, platform):
        """获取用户历史评论（平台过滤）"""
        reviews = self.interaction_tool.get_reviews(self.task['user_id'])
        platform_reviews = [r for r in reviews if r['source'] == platform]
        
        # 简化为最近5条评论
        recent_reviews = sorted(platform_reviews, key=lambda x: x['date'], reverse=True)[:5]
        return [f"{r['stars']}★: {r['text'][:100]}" for r in recent_reviews]

    def _build_prompt(self, platform, history, items):
        """构建平台特定Prompt"""
        platform_prompts = {
            'yelp': '''
                As a local guide, recommend businesses based on:
                1. User preferences from past reviews: {history}
                2. Business features: price range, parking, categories
                3. Rating and review count
                Rank these {count} local businesses:
                {items}
                Output ONLY the ranked IDs like ["id1","id2",...]
            ''',
            'amazon': '''
                As a shopping assistant, recommend products considering:
                1. User's purchase history: {history}
                2. Product brand, price, key features
                3. Rating and number of reviews
                Rank these {count} products:
                {items}
                Output ONLY the ranked IDs in ["id1","id2",...] format
            ''',
            'goodreads': '''
                As a librarian, recommend books based on:
                1. User's reading preferences: {history}
                2. Author reputation, book categories
                3. Average rating and popularity
                Rank these {count} books:
                {items}
                Output ONLY the ranked IDs as ["id1","id2",...]
            '''
        }
        
        item_str = '\n'.join([
            f"ID: {i['item_id']} | Name: {i['name']} | " + 
            ('Categories: ' + ', '.join(i.get('categories', [])) if platform=='yelp' else 
            'Features: ' + ', '.join(i.get('features', [])) if platform=='amazon' else
            'Author: ' + i.get('author', 'Unknown')) 
            for i in items
        ])
        
        return platform_prompts[platform].format(
            history=history,
            count=len(items),
            items=item_str
        )

    def _parse_result(self, text):
        """解析结果加强版"""
        try:
            # 尝试多种格式匹配
            matches = re.findall(r'"([^"]+)"', text) or re.findall(r"'([^']+)'", text)
            if len(matches) >= 20:
                return matches[:20]
            return self.task['candidate_list']  # 保底返回原始顺序
        except:
            return self.task['candidate_list']

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        # 获取平台类型（根据第一个候选商品判断）
        first_item = self.interaction_tool.get_item(self.task['candidate_list'][0])
        platform = first_item.get('source', 'yelp').lower()
        
        # 平台特定数据处理
        item_list = []
        for item_id in self.task['candidate_list']:
            raw_item = self.interaction_tool.get_item(item_id)
            processed_item = self._process_item(raw_item, platform)
            item_list.append(processed_item)
            
        # 获取用户历史评论（带平台过滤）
        history_review = self._get_user_history(platform)
        if not history_review:
            logging.warning(f"history reviews found for user {self.task['user_id']} on {platform}") 
        else:
            logging.info(f"history reviews found for user {self.task['user_id']} on {platform}")

        # 构建平台特定Prompt
        task_description = self._build_prompt(platform, history_review, item_list)
        
        # 调用LLM并解析结果
        result = self.reasoning(task_description)
        return self._parse_result(result)
    
    
        # plan = self.planning(task_type='Recommendation Task',
        #                      task_description="Please make a plan to query user information, you can choose to query user, item, and review information",
        #                      feedback='',
        #                      few_shot='')
        # print(f"The plan is :{plan}")
        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'}
         ]

        user = ''
        item_list = []
        history_review = ''
        for sub_task in plan:
            
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])

            elif 'item' in sub_task['description']:
                for n_bus in range(len(self.task['candidate_list'])):
                    item = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])
                    keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                item_list.append(filtered_item)
                # print(item)
            elif 'review' in sub_task['description']:
                history_review = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        task_description = f'''
        You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
        Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        Please rank the more interested items more front in your rank list.
        The information of the above 20 candidate items is as follows: {item_list}.

        Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        DO NOT output your analysis process!

        The correct output format:

        ['item id1', 'item id2', 'item id3', ...]

        '''
        result = self.reasoning(task_description)

        try:
            # print('Meta Output:',result)
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
            print('Processed Output:',eval(result))

            return eval(result)
        except:
            print('format error')
            return ['']

