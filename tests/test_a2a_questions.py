"""Questions in text (design §4 "Flow C", §7): an expert asks by pausing its task, and the person's
answer continues that same task. Seen only on the A2A wire and in what reaches the model."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from livekit.agents import Agent, AgentServer, InputRequired, RunContext, ToolResult
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.codec import metadata_from_wire
from livekit.agents.a2a.extension import KIND, KIND_DELEGATION, MERGED_INTO, VERBATIM
from livekit.agents.delegation import A2ADelegate
from livekit.agents.llm import ChatContext, ChatMessage, FunctionCallOutput
from livekit.agents.storage import AgentDBStorage

from .a2a_driver import (
    API_KEY,
    API_SECRET,
    EXPERT,
    Door,
    Reply,
    ScriptedLLM,
    call,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
    rpc,
    wire,
)

pytestmark = [pytest.mark.unit]

QUESTION = "What email address?"
DELEGATION = {KIND: KIND_DELEGATION}


def said(text: str) -> Callable[[ChatContext], bool]:
    """A request whose last item is the person's message ``text``."""

    def matches(ctx: ChatContext) -> bool:
        last = ctx.items[-1] if ctx.items else None
        return isinstance(last, ChatMessage) and last.role == "user" and last.text_content == text

    return matches


def asking(model: ScriptedLLM, *, verbatim: bool = True) -> Agent:
    """An agent whose ``book`` tool asks for an email until it has one."""
    agent = Agent(instructions="bookings", llm=model)

    @agent.tool
    async def book(ctx: RunContext, email: str = "") -> str | InputRequired:
        """Book the visit.

        Args:
            email: the person's email, when they gave it
        """
        if not email:
            return InputRequired(QUESTION, verbatim=verbatim)
        return f"booked for {email}"

    return agent


def runs(agent: Agent) -> Callable[[A2ASessionContext[Any]], Any]:
    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await agent.run(ctx.message, message_history=ctx.message_history)

    return handler


def user_texts(ctx: ChatContext) -> list[str]:
    return [
        i.text_content or "" for i in ctx.items if isinstance(i, ChatMessage) and i.role == "user"
    ]


def history_ids(task: dict[str, Any]) -> list[str]:
    return [m["messageId"] for m in task.get("history", [])]


@pytest.fixture
async def experts(agentdb_storage: AgentDBStorage) -> AsyncIterator[Callable[..., Expert]]:
    """``experts(handler)``: an expert's door on the same store, its client and a delegate."""
    made: list[Expert] = []

    def make(handler: Any) -> Expert:
        made.append(Expert(agentdb_storage, handler))
        return made[-1]

    yield make
    for e in made:
        await e.delegate.aclose()
        await e.http.aclose()


class Expert:
    def __init__(self, storage: AgentDBStorage, handler: Any) -> None:
        server = AgentServer(
            storage=storage, agent_name=EXPERT, api_key=API_KEY, api_secret=API_SECRET
        )
        server.a2a_session(endpoint=EXPERT, description="Books visits.")(handler)
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.http.app),
            base_url="http://experts.test",
            timeout=30.0,
        )
        self.delegate = A2ADelegate(f"http://experts.test/{EXPERT}", httpx_client=self.http)

    async def tasks(self, context_id: str) -> list[dict[str, Any]]:
        r = await self.http.post(
            f"/{EXPERT}/v1",
            json={
                "jsonrpc": "2.0",
                "id": "list",
                "method": "tasks/list",
                "params": {"contextId": context_id},
            },
        )
        return [t for t in r.json()["result"]["tasks"] if t.get("contextId") == context_id]


# ---------------------------------------------------------------------------------------------
# the question pauses the task; the answer continues it
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R31")
async def test_a_question_pauses_the_task_and_frees_the_lane(door: Callable[..., Door]) -> None:
    model = ScriptedLLM(Reply(tool_calls=[call("book", "b1")]), Reply("All quiet."))
    desk = door(runs(asking(model)))
    thread = await desk.thread()

    asked = await desk.send("book me in", context_id=thread)
    assert asked.state == "input_required"
    assert asked.status_text == QUESTION

    # nothing holds the lane while the task waits: another task runs at once
    other = await desk.send("anything new?", context_id=thread, **DELEGATION)
    assert other.task["id"] != asked.task["id"]
    assert other.state == "completed" and other.answer == "All quiet."


@pytest.mark.rule("R32", "R39")
@pytest.mark.xfail(
    strict=True,
    reason="R32, R39: Rust ask, continuation by taskId: today the next message answers the newest open ask by position (B15), and the answer overwrites the asking call's output",
)
async def test_the_answer_continues_the_waiting_task_as_one_record(
    door: Callable[..., Door],
) -> None:
    model = ScriptedLLM(
        Reply(tool_calls=[call("book", "b1")]),
        Reply(
            tool_calls=[call("book", "b2", email="ana@example.com")], when=said("ana@example.com")
        ),
        Reply("Booked.", when=output_of("b2")),
    )
    desk = door(runs(asking(model)))
    thread = await desk.thread()
    asked = await desk.send("book me in", context_id=thread, message_id="m1")
    task_id = asked.task["id"]

    answered = await desk.send(
        "ana@example.com", context_id=thread, task_id=task_id, message_id="m2"
    )
    assert {f["task"]["id"] for f in answered.frames if "task" in f} == {task_id}
    assert answered.task["id"] == task_id
    assert answered.state == "completed" and answered.answer == "Booked."
    assert MERGED_INTO not in answered.metadata
    assert history_ids(answered.task) == ["m1", "m2"]
    assert [t["id"] for t in await desk.list_tasks(thread)] == [task_id], "no row of its own"

    # one record: the asking call keeps its question, the answer is one message of the person
    request = model.requests[1]
    outputs = [i for i in request.items if isinstance(i, FunctionCallOutput)]
    assert [(o.call_id, QUESTION in str(o.output)) for o in outputs] == [("b1", True)]
    assert all(str(o.output) != "ana@example.com" for o in outputs)
    assert user_texts(request) == ["book me in", "ana@example.com"]


@pytest.mark.rule("R32")
@pytest.mark.xfail(
    strict=True,
    reason="R32: Rust ask, continuation by taskId: today the next message answers the newest open ask by position (B15)",
)
async def test_a_resent_answer_runs_nothing(door: Callable[..., Door]) -> None:
    model = ScriptedLLM(
        Reply(tool_calls=[call("book", "b1")]),
        Reply("Noted.", when=said("ana@example.com")),
    )
    desk = door(runs(asking(model)))
    thread = await desk.thread()
    asked = await desk.send("book me in", context_id=thread)
    task_id = asked.task["id"]
    first = await desk.send("ana@example.com", context_id=thread, task_id=task_id, message_id="m2")
    requests = len(model.requests)

    again = await desk.send("ana@example.com", context_id=thread, task_id=task_id, message_id="m2")
    assert again.task["id"] == task_id
    assert again.state == "completed" and again.answer == first.answer == "Noted."
    assert len(model.requests) == requests, "the resend ran no model"
    assert history_ids(again.task).count("m2") == 1
    assert len(await desk.list_tasks(thread)) == 1


# ---------------------------------------------------------------------------------------------
# what a message continues
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R34")
@pytest.mark.xfail(
    strict=True, reason="R34: the door does not route the person's message to the question in front"
)
async def test_the_persons_message_answers_the_question_in_front(
    door: Callable[..., Door],
) -> None:
    model = ScriptedLLM(
        Reply(tool_calls=[call("book", "b1")]),
        Reply("Noted.", when=said("ana@example.com")),
    )
    desk = door(runs(asking(model)))
    thread = await desk.thread()
    asked = await desk.send("book me in", context_id=thread)

    answered = await desk.send("ana@example.com", context_id=thread)
    assert answered.task["id"] == asked.task["id"]
    assert answered.state == "completed" and answered.answer == "Noted."
    assert len(await desk.list_tasks(thread)) == 1


@pytest.mark.rule("R34")
@pytest.mark.xfail(
    strict=True,
    reason="R34: Rust ask, continuation by taskId: a delegation answers the open question by position (B15)",
)
async def test_a_delegation_with_no_task_id_is_a_new_task(door: Callable[..., Door]) -> None:
    model = ScriptedLLM(
        Reply(tool_calls=[call("book", "b1")]),
        Reply("Here is the schedule.", when=said("what is open?")),
    )
    desk = door(runs(asking(model)))
    thread = await desk.thread()
    asked = await desk.send("book me in", context_id=thread)

    other = await desk.send("what is open?", context_id=thread, **DELEGATION)
    assert other.task["id"] != asked.task["id"]
    assert other.state == "completed" and other.answer == "Here is the schedule."
    still = await desk.get_task(asked.task["id"], thread)
    assert still["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"


@pytest.mark.rule("R35")
async def test_a_task_id_of_a_running_task_is_a_new_task(door: Callable[..., Door]) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        if ctx.message.text_content == "first":
            entered.set()
            await release.wait()
        return "reply to: " + (ctx.message.text_content or "")

    desk = door(handler, inbox="queue")
    thread = await desk.thread()
    first = asyncio.create_task(desk.send("first", context_id=thread))
    await entered.wait()
    running = next(t for t in await desk.list_tasks(thread))["id"]

    second = asyncio.create_task(desk.send("second", context_id=thread, task_id=running))
    while len(await desk.list_tasks(thread)) < 2:
        await asyncio.sleep(0.05)
    release.set()
    done_first, done_second = await asyncio.gather(first, second)

    assert done_first.task["id"] == running and done_first.answer == "reply to: first"
    assert {f["task"]["id"] for f in done_second.frames if "task" in f} != {running}
    assert done_second.task["id"] != running
    assert done_second.state == "completed" and done_second.answer == "reply to: second"


@pytest.mark.rule("R35")
@pytest.mark.xfail(
    strict=True, reason="R35: the door refuses a resend of the answer that ended the task"
)
async def test_a_task_id_of_an_ended_task_is_refused_but_a_resend_is_answered(
    door: Callable[..., Door],
) -> None:
    model = ScriptedLLM(
        Reply(tool_calls=[call("book", "b1")]),
        Reply("Noted.", when=said("ana@example.com")),
    )
    desk = door(runs(asking(model)))
    thread = await desk.thread()
    task_id = (await desk.send("book me in", context_id=thread)).task["id"]
    done = await desk.send("ana@example.com", context_id=thread, task_id=task_id, message_id="m2")
    assert done.state == "completed"

    later = wire("one more thing", "m3", context_id=thread)
    refused = await rpc(desk.http, "message/send", {"message": {**later, "taskId": task_id}})
    error = refused.json().get("error")
    assert error is not None and "completed" in error["message"]

    again = await desk.send("ana@example.com", context_id=thread, task_id=task_id, message_id="m2")
    assert again.task["id"] == task_id and again.answer == "Noted."


@pytest.mark.rule("R35")
async def test_a_task_id_of_another_lane_is_not_found(
    door: Callable[..., Door], experts: Callable[..., Expert]
) -> None:
    expert_model = ScriptedLLM(Reply(tool_calls=[call("book", "b1")]))
    expert = experts(runs(asking(expert_model)))
    desk = door()
    thread = await desk.thread()
    r = await expert.http.post(
        f"/{EXPERT}/v1",
        json={
            "jsonrpc": "2.0",
            "id": "send",
            "method": "message/send",
            "params": {"message": wire("book me in", "e1", context_id=thread, **DELEGATION)},
        },
    )
    expert_task = r.json()["result"]["task"]
    assert expert_task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"

    message = {**wire("ana@example.com", "m1", context_id=thread), "taskId": expert_task["id"]}
    error = (await rpc(desk.http, "message/send", {"message": message})).json().get("error")
    assert error is not None and error["code"] == -32001, "TaskNotFoundError"


@pytest.mark.rule("R52")
@pytest.mark.xfail(
    strict=True,
    reason="R52: the answer after the late question is a new task: no routing to the question in front (R34)",
)
async def test_a_message_taken_before_a_late_question_is_not_its_answer(
    door: Callable[..., Door],
) -> None:
    started, go = asyncio.Event(), asyncio.Event()
    model = ScriptedLLM(
        Reply(tool_calls=[call("slow_book", "b1")], when=said("book monday")),
        Reply("It is nine.", when=said("what time is it?")),
        Reply("Booked.", when=said("ana@example.com")),
    )
    agent = Agent(instructions="bookings", llm=model)

    @agent.tool
    async def slow_book(ctx: RunContext) -> InputRequired:
        """Book the visit."""
        started.set()
        await go.wait()
        return InputRequired(QUESTION, verbatim=True)

    desk = door(runs(agent), inbox="queue")
    thread = await desk.thread()
    first = asyncio.create_task(desk.send("book monday", context_id=thread))
    await started.wait()
    # the next message arrives while the question is still on its way: it waits for the lane
    handled = asyncio.create_task(desk.send("what time is it?", context_id=thread))
    while len(await desk.list_tasks(thread)) < 2:
        await asyncio.sleep(0.05)
    go.set()
    asked, handled_done = await asyncio.gather(first, handled)
    answer = await desk.send("ana@example.com", context_id=thread)

    assert asked.state == "input_required" and asked.status_text == QUESTION
    assert handled_done.task["id"] != asked.task["id"]
    assert handled_done.state == "completed" and handled_done.answer == "It is nine."
    assert answer.task["id"] == asked.task["id"], "the message after the question answers it"
    assert answer.state == "completed" and answer.answer == "Booked."
    tasks = await desk.list_tasks(thread)
    assert not any(MERGED_INTO in metadata_from_wire(t.get("metadata")) for t in tasks)


# ---------------------------------------------------------------------------------------------
# through a delegate
# ---------------------------------------------------------------------------------------------


def front_desk(delegate: A2ADelegate, model: ScriptedLLM) -> Agent:
    return Agent(instructions="front desk", tools=[delegate], llm=model)


@pytest.mark.rule("R40")
@pytest.mark.xfail(
    strict=True,
    reason="R40: the delegate sends the answer as a new expert task, which answers by position (B15)",
)
async def test_a_non_answer_goes_to_the_experts_model_in_the_same_task(
    door: Callable[..., Door], experts: Callable[..., Expert]
) -> None:
    expert_model = ScriptedLLM(
        Reply(tool_calls=[call("book", "e1")], when=said("book a visit")),
        # not an answer: the expert's model reads it, explains, and asks again
        Reply("For the confirmation.", tool_calls=[call("book", "e2")], when=said("why my email?")),
        Reply(
            tool_calls=[call("book", "e3", email="ana@example.com")], when=said("ana@example.com")
        ),
        Reply("Booked.", when=output_of("e3")),
    )
    expert = experts(runs(asking(expert_model)))
    desk_model = ScriptedLLM(
        Reply(tool_calls=[call("ask", "c1", request="book a visit")], when=said("book me in")),
        Reply("You are booked.", when=lambda ctx: "Booked." in str(ctx.items[-1])),
    )
    desk = door(runs(front_desk(expert.delegate, desk_model)))
    thread = await desk.thread()

    asked = await desk.send("book me in", context_id=thread)
    assert asked.state == "input_required" and asked.status_text == QUESTION

    async def expert_tasks() -> list[dict[str, Any]]:
        # the thread's tasks of every lane, but the front desk's
        return [t for t in await expert.tasks(thread) if t["id"] != asked.task["id"]]

    [expert_task] = [t["id"] for t in await expert_tasks()]

    again = await desk.send("why my email?", context_id=thread)
    assert again.task["id"] == asked.task["id"]
    assert again.state == "input_required"
    assert again.status_text is not None and QUESTION in again.status_text
    # the expert's own model read the person's words, in the same task
    assert user_texts(expert_model.requests[1])[-1] == "why my email?"
    [waiting] = await expert_tasks()
    assert waiting["id"] == expert_task
    assert waiting["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"

    done = await desk.send("ana@example.com", context_id=thread)
    assert done.task["id"] == asked.task["id"]
    assert done.state == "completed" and done.answer == "You are booked."
    assert [t["id"] for t in await expert_tasks()] == [expert_task]


# ---------------------------------------------------------------------------------------------
# the question carries what the turn said
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R43")
@pytest.mark.parametrize("before", ["", "Let me check."])
async def test_a_question_carries_what_the_turn_said_before_it(
    door: Callable[..., Door], before: str
) -> None:
    model = ScriptedLLM(Reply(before, tool_calls=[call("book", "b1")]))
    desk = door(runs(asking(model)))
    asked = await desk.send("book me in", context_id=await desk.thread())

    assert asked.state == "input_required"
    whole = f"{before}\n\n{QUESTION}" if before else QUESTION
    assert asked.status_text == whole
    # exact as a whole only when every part is
    assert (asked.metadata.get(VERBATIM) == whole) is (not before)


@pytest.mark.rule("R43", "B30")
async def test_an_experts_question_after_streamed_text_is_its_status_message(
    door: Callable[..., Door], experts: Callable[..., Expert]
) -> None:
    expert_model = ScriptedLLM(Reply("Let me check.", tool_calls=[call("book", "e1")]))
    expert = experts(runs(asking(expert_model)))
    desk_model = ScriptedLLM(
        Reply(tool_calls=[call("ask", "c1", request="book a visit")], when=said("book me in")),
        Reply("Checking now. Which email should I use?"),
    )
    desk = door(runs(front_desk(expert.delegate, desk_model)))
    asked = await desk.send("book me in", context_id=await desk.thread())

    assert asked.state == "input_required"
    assert asked.status_text == "Checking now. Which email should I use?"
    # the front desk phrased the whole question, not the text streamed before it
    phrased = desk_model.requests[-1].items[-1]
    assert isinstance(phrased, ChatMessage)
    assert "Let me check." in (phrased.text_content or "")
    assert QUESTION in (phrased.text_content or "")


@pytest.mark.rule("R43", "B31")
async def test_a_verbatim_result_before_an_experts_question_reaches_the_person_as_written(
    door: Callable[..., Door], experts: Callable[..., Expert]
) -> None:
    expert_model = ScriptedLLM(Reply(tool_calls=[call("policy", "e0"), call("book", "e1")]))
    expert_agent = asking(expert_model, verbatim=False)

    @expert_agent.tool
    async def policy(ctx: RunContext) -> ToolResult:
        """The booking policy."""
        return ToolResult("Visits cannot be refunded.", verbatim=True)

    expert = experts(runs(expert_agent))
    desk_model = ScriptedLLM(
        Reply(tool_calls=[call("ask", "c1", request="book a visit")], when=said("book me in")),
        Reply("Which email should I use?"),
    )
    desk = door(runs(front_desk(expert.delegate, desk_model)))
    asked = await desk.send("book me in", context_id=await desk.thread())

    assert asked.state == "input_required"
    assert asked.status_text == "Visits cannot be refunded.\n\nWhich email should I use?"
    assert asked.metadata.get(VERBATIM) == "Visits cannot be refunded."
    phrased = desk_model.requests[-1].items[-1]
    assert isinstance(phrased, ChatMessage)
    assert "Visits cannot be refunded." not in (phrased.text_content or "")
