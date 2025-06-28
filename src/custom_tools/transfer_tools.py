from langgraph.graph import END


class TransferTools:
    @staticmethod
    def transfer_to_director() -> str:
        """
        This is handoff helper function to transfer control to the director agent.

        Returns:
            str: The name of the director agent.
        """
        return "director"

    @staticmethod
    def transfer_to_writer() -> str:
        """
        Transfer handoff helper function to transfer control to the scenario writer agent.

        Returns:
            str: The name of the scenario writer agent.
        """
        return "writer"

    @staticmethod
    def inspector() -> str:
        """
        Transfer handoff helper function to transfer control to the inspector agent.

        Returns:
            str: The name of the inspector agent.
        """
        return "inspector"

    @staticmethod
    def end_graph() -> str:
        """
        End the graph execution.

        Returns:
            str: The END constant indicating the end of the graph.
        """
        return END
