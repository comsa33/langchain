"""Tests for verifying that testing utility code works as expected."""

from itertools import cycle
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import GenericFakeChatModel, ParrotFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from tests.unit_tests.stubs import AnyStr


def test_generic_fake_chat_model_invoke() -> None:
    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle([AIMessage(content="hello"), AIMessage(content="goodbye")])
    model = GenericFakeChatModel(messages=infinite_cycle)
    response = model.invoke("meow")
    assert response == AIMessage(content="hello", id=AnyStr())
    response = model.invoke("kitty")
    assert response == AIMessage(content="goodbye", id=AnyStr())
    response = model.invoke("meow")
    assert response == AIMessage(content="hello", id=AnyStr())


async def test_generic_fake_chat_model_ainvoke() -> None:
    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle([AIMessage(content="hello"), AIMessage(content="goodbye")])
    model = GenericFakeChatModel(messages=infinite_cycle)
    response = await model.ainvoke("meow")
    assert response == AIMessage(content="hello", id=AnyStr())
    response = await model.ainvoke("kitty")
    assert response == AIMessage(content="goodbye", id=AnyStr())
    response = await model.ainvoke("meow")
    assert response == AIMessage(content="hello", id=AnyStr())


async def test_generic_fake_chat_model_stream() -> None:
    """Test streaming."""
    infinite_cycle = cycle(
        [
            AIMessage(content="hello goodbye"),
        ]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)
    chunks = [chunk async for chunk in model.astream("meow")]
    assert chunks == [
        AIMessageChunk(content="hello", id=AnyStr()),
        AIMessageChunk(content=" ", id=AnyStr()),
        AIMessageChunk(content="goodbye", id=AnyStr()),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    chunks = [chunk for chunk in model.stream("meow")]
    assert chunks == [
        AIMessageChunk(content="hello", id=AnyStr()),
        AIMessageChunk(content=" ", id=AnyStr()),
        AIMessageChunk(content="goodbye", id=AnyStr()),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    # Test streaming of additional kwargs.
    # Relying on insertion order of the additional kwargs dict
    message = AIMessage(content="", additional_kwargs={"foo": 42, "bar": 24})
    model = GenericFakeChatModel(messages=cycle([message]))
    chunks = [chunk async for chunk in model.astream("meow")]
    assert chunks == [
        AIMessageChunk(content="", additional_kwargs={"foo": 42}, id=AnyStr()),
        AIMessageChunk(content="", additional_kwargs={"bar": 24}, id=AnyStr()),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "move_file",
                "arguments": '{\n  "source_path": "foo",\n  "'
                'destination_path": "bar"\n}',
            }
        },
    )
    model = GenericFakeChatModel(messages=cycle([message]))
    chunks = [chunk async for chunk in model.astream("meow")]

    assert chunks == [
        AIMessageChunk(
            content="",
            additional_kwargs={"function_call": {"name": "move_file"}},
            id=AnyStr(),
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": '{\n  "source_path": "foo"'},
            },
            id=AnyStr(),
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={"function_call": {"arguments": ","}},
            id=AnyStr(),
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": '\n  "destination_path": "bar"\n}'},
            },
            id=AnyStr(),
        ),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    accumulate_chunks = None
    for chunk in chunks:
        if accumulate_chunks is None:
            accumulate_chunks = chunk
        else:
            accumulate_chunks += chunk

    assert accumulate_chunks == AIMessageChunk(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "move_file",
                "arguments": '{\n  "source_path": "foo",\n  "'
                'destination_path": "bar"\n}',
            }
        },
        id=chunks[0].id,
    )


async def test_generic_fake_chat_model_astream_log() -> None:
    """Test streaming."""
    infinite_cycle = cycle([AIMessage(content="hello goodbye")])
    model = GenericFakeChatModel(messages=infinite_cycle)
    log_patches = [
        log_patch async for log_patch in model.astream_log("meow", diff=False)
    ]
    final = log_patches[-1]
    assert final.state["streamed_output"] == [
        AIMessageChunk(content="hello", id=AnyStr()),
        AIMessageChunk(content=" ", id=AnyStr()),
        AIMessageChunk(content="goodbye", id=AnyStr()),
    ]
    assert len({chunk.id for chunk in final.state["streamed_output"]}) == 1


async def test_callback_handlers() -> None:
    """Verify that model is implemented correctly with handlers working."""

    class MyCustomAsyncHandler(AsyncCallbackHandler):
        def __init__(self, store: List[str]) -> None:
            self.store = store

        async def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> Any:
            # Do nothing
            # Required to implement since this is an abstract method
            pass

        async def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> None:
            self.store.append(token)

    infinite_cycle = cycle(
        [
            AIMessage(content="hello goodbye"),
        ]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)
    tokens: List[str] = []
    # New model
    results = list(model.stream("meow", {"callbacks": [MyCustomAsyncHandler(tokens)]}))
    assert results == [
        AIMessageChunk(content="hello", id=AnyStr()),
        AIMessageChunk(content=" ", id=AnyStr()),
        AIMessageChunk(content="goodbye", id=AnyStr()),
    ]
    assert tokens == ["hello", " ", "goodbye"]
    assert len({chunk.id for chunk in results}) == 1


def test_chat_model_inputs() -> None:
    fake = ParrotFakeChatModel()

    assert fake.invoke("hello") == HumanMessage(content="hello", id=AnyStr())
    assert fake.invoke([("ai", "blah")]) == AIMessage(content="blah", id=AnyStr())
    assert fake.invoke([AIMessage(content="blah")]) == AIMessage(
        content="blah", id=AnyStr()
    )
