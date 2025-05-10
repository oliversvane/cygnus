from fastapi import APIRouter

router = APIRouter(prefix="/nodes", tags=["nodes"])

@router.get("/")
def get_nodes():
    """Get inofrmation about all nodes. Names, Versions, Last Heartbeats, Last Usages, Connection Status"""
    return {"message": "This is an example route"}

@router.get("/node")
def get_node():
    """Get inofrmation about a node. Name, Version, Last Heartbeat, Last Usage, Connection Status"""
    return {"message": "This is an example route"}

@router.post("/node")
def add_connection_to_node():
    """Add node to connection list"""
    return {"message": "This is an example route"}


@router.delete("/node")
def remove_connection_to_node():
    """Remove node from connection list"""
    return {"message": "This is an example route"}