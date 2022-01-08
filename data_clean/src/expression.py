import constant

class Expression(object):
    
    def __init__(self):
        # self.type is a string from ['And', 'For', 'When', 'Not', 'Null', 'Exists',
                                    #'Object', 'Has', 'Condition', 'Time', 'State',
                                    #'On', 'In', 'IsGrasping', 'IsNear', 'The']
        self.type = '' # self.type is a string
        self.arguments = [] # self.arguments is a list of Expression
        self.exp = '' # None except for leaves
        
        self.types_of = ['And', 'For', 'When', 'Not', 'Null', 'Exists',
                         'Object', 'Has', 'Condition', 'Time', 'State',
                         'On', 'In', 'IsGrasping', 'IsNear', 'The']
        
    def formExpression(self, parse): 
        """ Create the expression out of the string
            Expression: (and (Expression1) (Expression2) ... (ExpressionN)) -- And type
            Expression: (forall (Expression (Expression))) -- for
            Expression: (when (Expression) (Expression)) -- when
            Expression: (not (Expression)) -- not
            Expression: (predicate) -- base case
        """
        if (parse.find('(') == -1 and parse.find(')') == -1):
            self.type = 'Null'
            self.exp = parse
            raise Exception("I feel like dying")
        
        parse = parse.strip('\n')
        parse = parse.strip('\t')
        parse = parse.replace('\t',' ')
        parse = parse.replace('\n',' ')
        paren = parse.index('(')
        parse = parse[paren+1:]
        
        headers = ['and', 'forall', 'when', 'not', 'exists', 'object:t', 'has:t',
                   'condition:t', 'time:t', 'state:t', 'On:t', 'In:t', 'IsGrasping:t',
                   'IsNear:t', 'the:t']
        if parse.split(' ')[0] in headers:
            if parse.startswith('and'):
                self.type = 'And'
            elif parse.startswith('forall'):
                self.type = 'For'
            elif parse.startswith('when'):
                self.type = 'When'
            elif parse.startswith('not'):
                self.type = 'Not'
            elif parse.startswith('exists'):
                self.type = 'Exists'
            else:
                raise Exception(parse.split(' ')[0]+' unknown type')
            j = 0
            while ( j<len(parse) ):
                if (parse[j] == '('):
                    #create a new expression
                    exp_ = ''
                    exp_ = self.nextBracket(parse, j)
                    j = j + len(exp_) - 1
                    
                    exp1 = Expression()
                    exp1.formExpression(exp_)
                    
                    self.arguments.append(exp1)
                    
                elif (parse[j] == ' '):
                    # two cases occur _aasdfsafasdf_/) or _(expression)_
                    exp_ = ''
                    j = j+1
                    while (parse[j] != '(' and parse[j] != ' ' and parse[j] != ')'):
                        exp_ = exp_ + parse[j]
                        j = j+1
                        
                    if (len(exp_) != 0 and (parse[j] == ' ' or parse[j] == ')')):
                        exp1 = Expression()
                        exp1.formExpression(exp_)
                        self.arguments.append(exp1)
                        raise Exception('Should not have entered'+parse)
                    elif (parse[j] == '('):
                        j = j-1
                        
                j = j+1
            
        else:
            self.type = 'Null'
            if (parse[len(parse)-1] == ')'):
                parse = parse[:-1]
            open_ = 0
            close_ = 0
            index = -1
            for i in range(len(parse)):
                if parse[i] == '(':
                    open_ = open_ + 1
                elif parse[i] == ')':
                    close_ = close_ + 1
                    index = i
            if (open_ != close_ and index != -1):
                try:
                    parse = parse[0:index] + parse[index+1]
                except:
                    #parse = parse[0:index]
                    print('parse[index+1] out of range:'+parse)
            self.exp = parse.strip()
            
    def nextBracket(self, exp, start):
        # Function description: Given the start character as (, return the 
        # substring which ends with matching closing bracket
        if exp[start] != '(':
            raise Exception('Next Bracket: string must start with (')
        
        balance = 1
        ret = '('
        for i in range(start+1, len(exp)):
            ret = ret + exp[i]
            if exp[i] == '(':
                balance = balance+1
            if exp[i] == ')':
                balance = balance-1
                if balance == 0:
                    return ret
        return None
    
    def instantiate(self, expression, map):
        exp_ = str(expression)
        ret = ''
        words = exp_.split(' ')
        ret = words[0].strip()
        
        for i in range(1,len(words)):
            words[i] = words[i].strip()
            added = False
            for map_ in map:
                if map_[0] == words[i]:
                    added = True
                    ret = ret + ' '+map_[1]
            if not added:
                ret = ret + ' ' + words[i]
        return ret      
    
    def evaluate(self, env, map):
        if self.type == 'Null':
            instant = self.instantiate(self.exp, map)
            if len(instant) == 0:
                return ''
            code = env.isSatisfied(instant)
            if code == -1:
                return None
            elif code == 0:
                return instant
            elif code == 1:
                return ''
        elif self.type == 'And':
            ans = ''
            for exp in self.arguments:
                ret = exp.evaluate(env, map)
                if ret is None:
                    return None
                elif len(ret) > 0:
                    if len(ans) > 0:
                        ans = ret
                    else:
                        ans = ans+'^'
            return ans
        
        elif self.type == 'For':
            objL = env.getObj()
            resOuter = ''
            for obj in objL:
                map.append(['?otherobj', obj.getObjName()])
                resInner = self.arguments[1].evaluate(env, map)
                del map[-1]
                
                if resInner is None:
                    return None
                if resInner != '':
                    if len(resInner) > 0:
                        if len(resOuter) == 0:
                            resOuter = resInner
                        else:
                            resOuter = resOuter + '^' + resInner
            return resOuter
        
        elif self.type == 'When':
            condnEval = self.arguments[0].evaluate(env, map)
            if condnEval is None:
                return None
            if condnEval != '':
                return ''
            return self.arguments[1].evaluate(env, map)
        
        elif self.type == 'Not':
            res = self.arguments[0].evaluate(env, map)
            if res is None:
                return None
            if res == '':
                return 'not ('+self.instantiate(self.arguments[0].exp, map) + ')'
            else:
                return ''
        else:
            raise Exception('Unknown PDDL syntax '+self.exp)
        
        return None
    def modify(self, env, map):
        if self.type == 'Null':
            instant = self.instantiate(self.exp, map)
            if not instant == '':
                env.modify(instant, True)
        elif self.type == 'And':
            """Handling if-else - a common scenario occurs as
               (and (when (C) (not C)) (when (not C) (C)))
               (and (when (not C) (C)) (when (C) (not C)))
            """
            sw = open(constant.root_dir + 'and.txt', 'w')
            allowed = [] # allowed is a list of int
            for i in range(len(self.arguments)):
                if self.arguments[i].type == 'When':
                    ans = self.arguments[i].arguments[0].evaluate(env, map)
                    if ans is not None and ans == '':
                        if self.arguments[i].arguments[0].type == 'Null':
                            sw.write("Allowing: "+self.arguments[i].arguments[0].exp)
                        allowed.append(i)
                else:
                    allowed.append(i)
            sw.close()
            
            for i in range(len(allowed)):
                exp = self.arguments[allowed[i]]
                exp.modify(env, map)
            
        elif self.type == 'For':
            objL = env.getObj()
            for obj in objL:
                map.append(['?otherobj', obj.getObjName()])
                self.arguments[1].modify(env, map)
                del map[-1]
        elif self.type == 'When':
            condnEval = self.arguments[0].evaluate(env, map)
            if condnEval is not None and condnEval == '':
                self.arguments[1].modify(env, map)
        elif self.type == 'Not':
            innerAtom = self.arguments[0]
            if innerAtom.type != 'Null':
                raise Exception('Non-deterministic Effect Not (P and Q)')
            instant = self.instantiate(innerAtom.exp, map)
            if instant == '':
                raise Exception('')
            env.modify(instant, False)
        else:
            raise Exception('Unknown PDDL syntax '+self.exp)       
            
            
            
            
            
            
            
            
            
            
            
            
            
            