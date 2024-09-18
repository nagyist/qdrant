import multiprocessing
import pathlib
import random
from time import sleep
from typing import Any, Literal

from .test_dummy_shard import assert_http_response

from .fixtures import upsert_random_points, create_collection, random_dense_vector
from .utils import *


COLLECTION_NAME = "test_collection"


def test_resharding_state_transitions(tmp_path: pathlib.Path):
    # Bootstrap resharding cluster
    peer_uris, _ = bootstrap_resharding(tmp_path)

    # Check that we can't (re)apply unexpected resharding state transitions
    try_requests(peer_uris[0], 400, [
        start_resharding,
        lambda peer_uri: start_resharding(peer_uri, direction="down"),
        commit_write_hashring,
        finish_resharding,
    ])

    # Commit read hashring
    resp = commit_read_hashring(peer_uris[0])
    assert_http_ok(resp)

    # Check that we can't (re)apply unexpected resharding state transitions or abort resharding
    try_requests(peer_uris[0], 400, [
        start_resharding,
        lambda peer_uri: start_resharding(peer_uri, direction="down"),
        commit_read_hashring,
        finish_resharding,
        abort_resharding,
    ])

    # Commit write hashring
    resp = commit_write_hashring(peer_uris[0])
    assert_http_ok(resp)

    # Check that we can't (re)apply unexpected resharding state transitions or abort resharding
    try_requests(peer_uris[0], 400, [
        start_resharding,
        lambda peer_uri: start_resharding(peer_uri, direction="down"),
        commit_read_hashring,
        commit_write_hashring,
        abort_resharding,
    ])

    # Finish resharding
    resp = finish_resharding(peer_uris[0])
    assert_http_ok(resp)

    # Wait for resharding to finish
    wait_for_collection_resharding_operations_count(peer_uris[0], COLLECTION_NAME, 0)

def test_resharding_abort(tmp_path: pathlib.Path):
    # Bootstrap resharding cluster
    peer_uris, _ = bootstrap_resharding(tmp_path)

    # Abort resharding
    resp = abort_resharding(peer_uris[0])
    assert_http_ok(resp)

    # Wait for resharding to abort
    wait_for_collection_resharding_operations_count(peer_uris[0], COLLECTION_NAME, 0)

def test_resharding_abort_on_delete_collection(tmp_path: pathlib.Path):
    # Bootstrap resharding cluster
    peer_uris, peer_ids = bootstrap_resharding(tmp_path, peer_idx=-1)

    # Remove target peer
    resp = requests.delete(f"{peer_uris[0]}/collections/{COLLECTION_NAME}")
    assert_http_ok(resp)

    # TODO: Wait for/check... *what*? 🤔

def test_resharding_abort_on_delete_shard_key(tmp_path: pathlib.Path):
    # Bootstrap resharding cluster
    peer_uris, peer_ids = bootstrap_resharding(
        tmp_path,
        shard_keys=["custom_shard_key_1", "custom_shard_key_2"],
        shard_key="custom_shard_key_2",
    )

    # Delete shard key
    resp = requests.post(f"{peer_uris[0]}/collections/{COLLECTION_NAME}/shards/delete", json={
        "shard_key": "custom_shard_key_2",
    })

    assert_http_ok(resp)

    # Wait for resharding to abort
    wait_for_collection_resharding_operations_count(peer_uris[0], COLLECTION_NAME, 0)

def test_resharding_abort_on_remove_peer(tmp_path: pathlib.Path):
    # Bootstrap resharding cluster
    peer_uris, peer_ids = bootstrap_resharding(tmp_path, peer_idx=-1)

    # Remove target peer
    resp = requests.delete(f"{peer_uris[0]}/cluster/peer/{peer_ids[-1]}?force=true")
    assert_http_ok(resp)

    # Wait for resharding to abort
    wait_for_collection_resharding_operations_count(peer_uris[0], COLLECTION_NAME, 0)

def test_resharding_try_remove_target_shard(tmp_path: pathlib.Path):
    # Bootstrap resharding cluster
    peer_uris, peer_ids = bootstrap_resharding(tmp_path)

    # Try to remove target shard
    resp = requests.post(f"{peer_uris[0]}/collections/{COLLECTION_NAME}/cluster", json={
        "drop_replica": {
            "peer_id": peer_ids[0],
            "shard_id": 3,
        }
    })

    assert_http(resp, 400)

@pytest.mark.parametrize("direction", [("up"), ("down")])
def test_resharding_forward(tmp_path: pathlib.Path, direction: Literal["up", "down"]):
    # Bootstrap resharding cluster
    peer_uris, peer_ids = bootstrap_resharding(tmp_path, direction=direction)

    # Upsert points to collection
    upsert_random_points(peer_uris[0], 1000, collection_name=COLLECTION_NAME)

    # Select shards for the test
    shard_id, to_shard_id = select_resharding_shards(3, direction)

    # Find peers for the test
    info = get_collection_cluster_info(peer_uris[0], COLLECTION_NAME)
    from_peer_id, to_peer_id = find_replicas(info, [shard_id, to_shard_id])

    # Get peer URIs
    from_peer_uri = peer_uris[peer_ids.index(from_peer_id)]
    to_peer_uri = peer_uris[peer_ids.index(to_peer_id)]

    # Assert that all points were correctly forwarded
    offset = 0

    while offset is not None:
        from_resp = scroll_local_points(from_peer_uri, shard_id, to_shard_id, offset, 1000)
        to_resp = scroll_local_points(to_peer_uri, to_shard_id, to_shard_id, offset, 1000)

        assert from_resp['points'] == to_resp['points']
        assert from_resp['next_page_offset'] == to_resp['next_page_offset']

        offset = from_resp['next_page_offset']

@pytest.mark.parametrize("direction", [("up"), ("down")])
def test_resharding_transfer(tmp_path: pathlib.Path, direction: Literal["up", "down"]):
    # Bootstrap resharding cluster
    peer_uris, peer_ids = bootstrap_resharding(tmp_path, upsert_points=1000, direction=direction)

    # Select shards for resharding transfer
    shard_id, to_shard_id = select_resharding_shards(3, direction)

    # Migrate points
    from_peer_id, to_peer_id = migrate_points(peer_uris[0], shard_id, to_shard_id)

    # Get peer URIs
    from_peer_uri = peer_uris[peer_ids.index(from_peer_id)]
    to_peer_uri = peer_uris[peer_ids.index(to_peer_id)]

    # Assert that all points were correctly migrated
    offset = 0

    while offset is not None:
        from_resp = scroll_local_points(from_peer_uri, shard_id, to_shard_id, offset, 1000)
        to_resp = scroll_local_points(to_peer_uri, to_shard_id, to_shard_id, offset, 1000)

        assert from_resp['points'] == to_resp['points']
        assert from_resp['next_page_offset'] == to_resp['next_page_offset']

        offset = from_resp['next_page_offset']

def test_resharding_down_abort_cleanup(tmp_path: pathlib.Path):
    # Bootstrap resharding cluster
    peer_uris, peer_ids = bootstrap_resharding(tmp_path, upsert_points=1000, direction="down")

    # Migrate points from shard 2 to shard 0
    _, to_peer_id = migrate_points(peer_uris[0], 2, 0)

    # Get target peer URI
    to_peer_uri = peer_uris[peer_ids.index(to_peer_id)]

    # Assert that some points were migrated to target peer
    resp = scroll_local_points(to_peer_uri, 0, 2)
    assert len(resp['points']) > 0

    # Abort resharding
    resp = abort_resharding(peer_uris[0])
    assert_http_ok(resp)

    # Wait for resharding to abort
    wait_for_collection_resharding_operations_count(peer_uris[0], COLLECTION_NAME, 0)

    # Assert that all replicas are in `Active` state
    info = get_collection_cluster_info(peer_uris[0], COLLECTION_NAME)

    for replica in all_replicas(info):
        assert replica["state"] == "Active"

    # Assert that migrated points were deleted from target peer
    resp = requests.post(f"{to_peer_uri}/collections/{COLLECTION_NAME}/shards/0/points/scroll", json={
        "hash_ring_filter": {
            "expected_shard_id": 2,
        }
    })

    assert_http_ok(resp)

    points = resp.json()['result']['points']
    assert len(points) == 0


def bootstrap_resharding(
    tmp_path: pathlib.Path,
    collection: str = COLLECTION_NAME,
    peer_idx: int | None = None,
    **kwargs,
):
    # Bootstrap cluster
    peer_uris, peer_ids = bootstrap_cluster(tmp_path, collection, **kwargs)

    # Select target peer
    peer_id = None

    if peer_idx:
        try:
            peer_id = peer_ids[peer_idx]
        finally:
            pass

    # Start resharding
    resp = start_resharding(peer_uris[0], collection, peer_id=peer_id, **kwargs)
    assert_http_ok(resp)

    # Wait for resharding to start
    wait_for_collection_resharding_operations_count(peer_uris[0], collection, 1)

    return (peer_uris, peer_ids)

def bootstrap_cluster(
    tmp_path: pathlib.Path,
    collection: str = COLLECTION_NAME,
    shard_number: int = 3,
    replication_factor: int = 2,
    shard_keys: list[str] | str | None = None,
    upsert_points: int = 0,
    peers: int = 3,
    **kwargs,
) -> tuple[list[str], list[int]]:
    assert_project_root()

    # Prevent optimizers messing with point counts
    env = {
        "QDRANT__STORAGE__OPTIMIZERS__INDEXING_THRESHOLD_KB": "0",
    }

    # Start cluster
    peer_uris, _, _ = start_cluster(tmp_path, peers, extra_env=env)

    # Collect peer IDs
    peer_ids = []
    for peer_uri in peer_uris:
        peer_ids.append(get_cluster_info(peer_uri)["peer_id"])

    # Create collection
    create_collection(
        peer_uris[0],
        collection,
        shard_number,
        replication_factor,
        sharding_method="auto" if shard_keys is None else "custom",
    )

    wait_collection_exists_and_active_on_all_peers(collection, peer_uris)

    # Create custom shard keys (if required), and upload points to collection
    if type(shard_keys) is not list:
        shard_keys: list[str | None] = [shard_keys]

    for shard_key in shard_keys:
        # Create custom shard key (if required)
        if shard_key is not None:
            resp = requests.put(f"{peer_uris[0]}/collections/{collection}/shards", json={
                "shard_key": shard_key,
                "shards_number": shard_number,
                "replication_factor": replication_factor,
            })

            assert_http_ok(resp)

        # Upsert points to collection
        if upsert_points > 0:
            upsert_random_points(
                peer_uris[0],
                upsert_points,
                collection_name=collection,
                shard_key=shard_key,
            )

    return (peer_uris, peer_ids)


def start_resharding(
    peer_uri: str,
    collection: str = COLLECTION_NAME,
    direction: Literal["up", "down"] = "up",
    peer_id: int | None = None,
    shard_key: str | None = None,
    **kwargs,
):
    return requests.post(f"{peer_uri}/collections/{collection}/cluster", json={
        "start_resharding": {
            "direction": direction,
            "peer_id": peer_id,
            "shard_key": shard_key,
        }
    })

def commit_read_hashring(peer_uri: str, collection: str = COLLECTION_NAME):
    return requests.post(f"{peer_uri}/collections/{collection}/cluster", json={
        "commit_read_hash_ring": {}
    })

def commit_write_hashring(peer_uri: str, collection: str = COLLECTION_NAME):
    return requests.post(f"{peer_uri}/collections/{collection}/cluster", json={
        "commit_write_hash_ring": {}
    })

def finish_resharding(peer_uri: str, collection: str = COLLECTION_NAME):
    return requests.post(f"{peer_uri}/collections/{collection}/cluster", json={
        "finish_resharding": {}
    })

def abort_resharding(peer_uri: str, collection: str = COLLECTION_NAME):
    return requests.post(f"{peer_uri}/collections/{collection}/cluster", json={
        "abort_resharding": {}
    })


def scroll_local_points(
    peer_uri: str,
    shard_id: int,
    filter_shard_id: int | None = None,
    offset: int | None = None,
    limit: int | None = None,
    collection: str = COLLECTION_NAME,
):
    resp = requests.post(f"{peer_uri}/collections/{collection}/shards/{shard_id}/points/scroll", json={
        "limit": limit,
        "offset": offset,
        "hash_ring_filter": None if filter_shard_id is None else {
            "expected_shard_id": filter_shard_id,
        }
    })

    assert_http_ok(resp)

    return resp.json()['result']


def select_resharding_shards(peers: int, direction: Literal["up", "down"]):
    if direction == "up":
        return (0, peers)
    elif direction == "down":
        assert peers > 1
        return (peers - 1, 0)
    else:
        raise Exception(f"invalid resharding direction {direction}")

def migrate_points(peer_uri: str, shard_id: int, to_shard_id: int, collection: str = COLLECTION_NAME) -> tuple[int, int]:
    # Find peers for resharding transfer
    info = get_collection_cluster_info(peer_uri, collection)
    from_peer_id, to_peer_id = find_replicas(info, [shard_id, to_shard_id])

    # Start resharding transfer
    resp = requests.post(f"{peer_uri}/collections/{collection}/cluster", json={
        "replicate_shard": {
            "from_peer_id": from_peer_id,
            "to_peer_id": to_peer_id,
            "shard_id": shard_id,
            "to_shard_id": to_shard_id,
            "method": "resharding_stream_records",
        }
    })

    assert_http_ok(resp)

    # Wait for resharding trasnfer to start
    sleep(1)

    # Wait for resharding transfer to finish or abort
    wait_for_collection_shard_transfers_count(peer_uri, collection, 0)

    # Assert that resharding transfer finished successfully
    info = get_collection_cluster_info(peer_uri, collection)

    # Assert that resharding is still in progress
    assert "resharding_operations" in info and len(info["resharding_operations"]) > 0

    # Assert that `to_peer_id`/`to_shard_id` replica is in `Resharding` state
    migration_successful = False

    for replica in all_replicas(info):
        if replica["peer_id"] == to_peer_id and replica["shard_id"] == to_shard_id and replica["state"] == "Resharding":
            migration_successful = True
            break

    assert migration_successful

    # Return target peer
    return (from_peer_id, to_peer_id)

def find_replicas(info: dict[str, Any], shard_ids: list[int]) -> list[int]:
    peers = [None for _ in range(len(shard_ids))]

    for replica in all_replicas(info):
        for idx, shard_id in enumerate(shard_ids):
            if replica["shard_id"] == shard_id:
                peers[idx] = peers[idx] or replica["peer_id"]

    assert None not in peers

    return peers

def all_replicas(info: dict[str, Any]):
    for local in info["local_shards"]:
        local["peer_id"] = info["peer_id"]
        yield local

    for remote in info["remote_shards"]:
        yield remote


def try_requests(
    peer_uri: str,
    expected_status: int,
    reqs: list[Callable[[str], requests.Response]],
):
    for req in reqs:
        resp = req(peer_uri)
        assert_http(resp, expected_status)

def assert_http(resp: requests.Response, expected_status: int):
    assert resp.status_code == expected_status, (
        f"`{resp.url}` "
        f"returned an unexpected status code (expected {expected_status}, received {resp.status_code}):\n"
        f"{resp.json()}"
    )


def wait_for_one_of_resharding_operation_stages(
    peer_uri: str, expected_stages: list[str], **kwargs
):
    def resharding_operation_stages():
        requests.post(f"{peer_uri}/collections/{COLLECTION_NAME}/points/scroll")

        info = get_collection_cluster_info(peer_uri, COLLECTION_NAME)

        if "resharding_operations" not in info:
            return False

        for resharding in info["resharding_operations"]:
            if not "comment" in resharding:
                continue

            stage, *_ = resharding["comment"].split(":", maxsplit=1)

            if stage in expected_stages:
                return True

        return False

    wait_for(resharding_operation_stages, **kwargs)

def wait_for_resharding_shard_transfer_info(
    peer_uri: str, expected_stage: str | None, expected_method: str
):
    if expected_stage is not None:
        wait_for_collection_resharding_operation_stage(
            peer_uri, COLLECTION_NAME, expected_stage
        )

    wait_for_collection_shard_transfer_method(
        peer_uri, COLLECTION_NAME, expected_method
    )

    info = get_collection_cluster_info(peer_uri, COLLECTION_NAME)
    return info["shard_transfers"][0]

def wait_for_resharding_to_finish(peer_uris: list[str], expected_shard_number: int):
    # Wait for resharding to finish
    for peer_uri in peer_uris:
        wait_for_collection_resharding_operations_count(
            peer_uri,
            COLLECTION_NAME,
            0,
            wait_for_timeout=60,
        )

    # Check number of shards in the collection
    for peer_uri in peer_uris:
        resp = get_collection_cluster_info(peer_uri, COLLECTION_NAME)
        assert resp["shard_count"] == expected_shard_number
