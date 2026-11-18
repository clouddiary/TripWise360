"""Offline graph tests: real LangGraph checkpoints/interrupts, mocked providers."""
import ast
import json
import operator
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, Annotated

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from workflow_rules import GuardrailDecision, HumanReview, RevisedRequest


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path('backend.py').read_text())
        ns = dict(Any=Any, TypedDict=TypedDict, Annotated=Annotated, operator=operator,
                  AnyMessage=AnyMessage, AIMessage=AIMessage, HumanMessage=HumanMessage,
                  SystemMessage=SystemMessage, json=json, StateGraph=StateGraph,
                  START=START, END=END, interrupt=interrupt, Command=Command,
                  GuardrailDecision=GuardrailDecision, HumanReview=HumanReview,
                  RevisedRequest=RevisedRequest)
        for name in ('log_agent_start', 'log_agent_done', 'log_state_read', 'log_state_write',
                     'log_llm_call', 'log_llm_done', 'log_error', 'log_result', 'log_system', 'log_graph_done'):
            ns[name] = lambda *args, **kwargs: None
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        definitions += [n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ('KNOWN_AGENTS', 'AGENT_ORDER', 'ROUTE_MAP') for t in n.targets)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'backend.py', 'exec'), ns)
        self.ns = ns
        self.responses = []
        self.calls = []
        def model(*args):
            value = self.responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        ns['_llm_text'] = model
        def final_model(messages):
            self.calls.append('final')
            return AIMessage(content='Approved final response')
        ns['llm'] = SimpleNamespace(invoke=final_model)
        for agent, field in [('flight_agent', 'flight_results'), ('hotel_agent', 'hotel_results'),
                             ('weather_agent', 'weather_results'), ('budget_agent', 'budget_results'),
                             ('itinerary_agent', 'itinerary')]:
            def specialist(state, name=agent, key=field):
                self.calls.append((name, state['user_query']))
                return {key: f"{name}: {state['user_query']}", 'approval_request': 'Review draft'}
            ns[agent] = specialist
        start = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='graph' for t in n.targets))
        end = next(i for i,n in enumerate(tree.body[start:], start) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATABASE_URL' for t in n.targets))
        exec(compile(ast.Module(body=tree.body[start:end],type_ignores=[]),'backend.py','exec'),ns)
        self.saver = InMemorySaver()
        self.graph = ns['graph'].compile(checkpointer=self.saver)
        ns['travel_graph'] = self.graph
        self.config = {'configurable': {'thread_id': self.id()}}

    def plan(self, agents):
        return json.dumps({'selected_agents': agents, 'trip_constraints': {}, 'reasoning': 'Requested scope'})

    def test_revise_research_review_then_approve(self):
        self.responses = ['{"allowed":true,"reason":"Travel"}', self.plan(['hotel_agent'])]
        initial = self.graph.invoke({'user_query':'Paris for 4 days', 'llm_calls':0}, self.config)
        self.assertIn('__interrupt__', initial)
        self.responses = [json.dumps({'updated_request':'Rome for 5 days; no hotels; check weather'}),
                          '{"allowed":true,"reason":"Travel"}', self.plan(['weather_agent'])]
        revised = self.graph.invoke(Command(resume={'approved':False,'feedback':'Make it Rome for five days. No hotels; check weather.'}),self.config)
        self.assertIn('__interrupt__', revised)
        self.assertEqual(revised['revision_count'],1)
        self.assertEqual(revised['hotel_results'],'')
        self.assertIn('Rome for 5 days',revised['weather_results'])
        self.assertNotIn('final',self.calls)
        final = self.graph.invoke(Command(resume={'approved':True,'feedback':''}),self.config)
        self.assertNotIn('__interrupt__',final)
        self.assertEqual(final['final_response'],'Approved final response')
        self.assertEqual(self.calls.count('final'),1)

    def test_multiple_revisions_each_require_new_approval(self):
        self.responses=['{"allowed":true,"reason":"Travel"}',self.plan([])]
        self.graph.invoke({'user_query':'Paris for 4 days'},self.config)
        for duration in (5,6):
            self.responses=[json.dumps({'updated_request':f'Paris for {duration} days'}),
                            '{"allowed":true,"reason":"Travel"}',self.plan([])]
            result=self.graph.invoke(Command(resume={'approved':False,'feedback':'Add one day'}),self.config)
            self.assertIn('__interrupt__',result)
            self.assertEqual(result['user_query'],f'Paris for {duration} days')
            self.assertEqual(result['revision_count'],duration-4)
            self.assertNotIn('final',self.calls)

    def test_http_review_contract_rejects_string_boolean(self):
        tree=ast.parse(Path('app.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ApprovalRequest')
        from pydantic import Field
        ns={'HumanReview':HumanReview,'Field':Field}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'app.py','exec'),ns)
        with self.assertRaises(ValueError):
            ns['ApprovalRequest'](thread_id='trip',approved='false',feedback='Change')
        self.assertIs(ns['ApprovalRequest'](thread_id='trip',approved=True).approved,True)

    def test_guardrail_invalid_responses_block_before_research(self):
        for raw in ['{"allowed":"false","reason":"No"}', '{}',
                    '{"allowed":1,"reason":"No"}', '{"allowed":null,"reason":"No"}',
                    '{"allowed":true,"reason":"OK","extra":1}',
                    '{"allowed":false,"reason":""}', '```json\n{"allowed":true,"reason":"OK"}\n```',
                    'not json', RuntimeError('provider offline')]:
            with self.subTest(raw=raw):
                self.responses=[raw]
                decision=self.ns['supervisor_agent']({'user_query':'Test'})
                self.assertIs(decision['guardrail_allowed'],False)
                self.assertEqual(decision['selected_agents'],[])
        self.assertEqual(self.calls,[])

    def test_revision_is_guardrailed_again(self):
        self.responses=['{"allowed":true,"reason":"Travel"}',self.plan([])]
        self.graph.invoke({'user_query':'Paris for 4 days'},self.config)
        self.responses=['{"updated_request":"Unsafe revised request"}','{"allowed":false,"reason":"Unsafe"}']
        result=self.graph.invoke(Command(resume={'approved':False,'feedback':'Unsafe change'}),self.config)
        self.assertIs(result['guardrail_allowed'],False)
        self.assertNotIn('__interrupt__',result)
        self.assertNotIn('final',self.calls)

    def test_revision_failure_returns_to_review_without_finalizing(self):
        self.responses=['{"allowed":true,"reason":"Travel"}',self.plan([])]
        first=self.graph.invoke({'user_query':'Paris for 4 days'},self.config)
        self.responses=['bad JSON']
        result=self.graph.invoke(Command(resume={'approved':False,'feedback':'Add a day'}),self.config)
        self.assertIn('__interrupt__',result)
        self.assertEqual(result['itinerary'],first['itinerary'])
        self.assertTrue(result['revision_error'])
        self.assertNotIn('final',self.calls)

    def test_strict_human_review(self):
        for value in [{'approved':'false','feedback':'Change'}, {'approved':False,'feedback':' '}, {'approved':1}]:
            with self.assertRaises(ValueError):
                HumanReview.model_validate(value)

    def test_load_pending_thread_after_graph_restart_then_approve(self):
        self.responses=['{"allowed":true,"reason":"Travel"}',self.plan([])]
        original=self.graph.invoke({'user_query':'Paris for 4 days'},self.config)
        calls_before=list(self.calls)
        self.ns['travel_graph']=self.ns['graph'].compile(checkpointer=self.saver)
        loaded=self.ns['load_travel_thread'](self.id())
        self.assertTrue(loaded['requires_approval'])
        self.assertEqual(loaded['itinerary'],original['itinerary'])
        self.assertEqual(loaded['user_query'],'Paris for 4 days')
        self.assertEqual(self.calls,calls_before)
        completed=self.ns['resume_travel_agent'](self.id(),True)
        self.assertFalse(completed['requires_approval'])
        self.assertEqual(completed['answer'],'Approved final response')
        loaded_final=self.ns['load_travel_thread'](self.id())
        self.assertFalse(loaded_final['requires_approval'])
        self.assertEqual(loaded_final['answer'],'Approved final response')

    def test_load_unknown_thread_does_not_start_graph(self):
        with self.assertRaises(LookupError):
            self.ns['load_travel_thread']('unknown')
        self.assertEqual(self.calls,[])

    def test_resume_requires_pending_review(self):
        with self.assertRaisesRegex(ValueError,'not waiting'):
            self.ns['resume_travel_agent']('missing',True)

if __name__ == '__main__':
    unittest.main()
