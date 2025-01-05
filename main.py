import qrcode
from build123d import BuildPart, Box, Location, export_gltf, Compound, Mesher, Align
from ocp_vscode import show_object
import numpy as np
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import time


class QRCode3DGenerator:
    def __init__(
        self,
        text: str,
        block_size: float = 5.0,
        block_height: float = 4.0,
        support_width: float = 0.8,
        diagonal_support_width: float = 1.2,
        version: int = 1,
        error_correction: int = qrcode.constants.ERROR_CORRECT_L,
    ):
        self.text = text
        self.block_size = block_size
        self.block_height = block_height
        self.support_width = support_width
        self.diagonal_support_width = diagonal_support_width
        self.version = version
        self.error_correction = error_correction
        self.matrix = self._generate_qr_matrix()
        self.rows = len(self.matrix)
        self.cols = len(self.matrix[0])
        self.connected_components = self._find_connected_components()

    def _generate_qr_matrix(self) -> List[List[bool]]:
        """Generate the QR code matrix."""
        qr = qrcode.QRCode(
            version=self.version,
            error_correction=self.error_correction,
            box_size=1,
            border=4,
        )
        qr.add_data(self.text)
        qr.make(fit=True)
        return qr.modules

    def _find_connected_components(self) -> Dict[Tuple[int, int], int]:
        """Find all connected components in the QR code using Union-Find algorithm."""

        def find(parent, x):
            if parent[x] != x:
                parent[x] = find(parent, parent[x])
            return parent[x]

        def union(parent, rank, x, y):
            px, py = find(parent, x), find(parent, y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        components = {}
        parent = {}
        rank = defaultdict(int)

        for i in range(self.rows):
            for j in range(self.cols):
                if self.matrix[i][j]:
                    components[(i, j)] = len(parent)
                    parent[(i, j)] = (i, j)

        # Connect adjacent modules (only orthogonal connections for components)
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for i in range(self.rows):
            for j in range(self.cols):
                if not self.matrix[i][j]:
                    continue
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < self.rows
                        and 0 <= nj < self.cols
                        and self.matrix[ni][nj]
                    ):
                        union(parent, rank, (i, j), (ni, nj))

        component_mapping = {}
        for pos in components:
            component_mapping[pos] = find(parent, pos)

        return component_mapping

    def _needs_diagonal_support(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Check if a module needs diagonal supports and return the positions needing support."""
        if not self.matrix[row][col]:
            return []

        diagonal_supports = []
        diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for di, dj in diagonals:
            ni, nj = row + di, col + dj
            if 0 <= ni < self.rows and 0 <= nj < self.cols and self.matrix[ni][nj]:
                # Check if there's no orthogonal connection
                has_orthogonal = False
                if self.matrix[row][nj] or self.matrix[ni][col]:
                    has_orthogonal = True

                if not has_orthogonal:
                    diagonal_supports.append((ni, nj))

        return diagonal_supports

    def _find_nearest_neighbors(
        self, component_id: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Find the nearest neighboring components for a given component."""
        component_positions = [
            pos
            for pos, comp in self.connected_components.items()
            if comp == component_id
        ]
        other_components = set(self.connected_components.values()) - {component_id}

        nearest_neighbors = []
        min_distance = float("inf")

        for other_comp in other_components:
            other_positions = [
                pos
                for pos, comp in self.connected_components.items()
                if comp == other_comp
            ]

            for pos1 in component_positions:
                for pos2 in other_positions:
                    dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
                    if dist < min_distance:
                        min_distance = dist
                        nearest_neighbors = [(pos1, pos2)]
                    elif dist == min_distance:
                        nearest_neighbors.append((pos1, pos2))

        return nearest_neighbors

    def _create_support(self, start: Tuple[int, int], end: Tuple[int, int]) -> Box:
        """Create a support structure between two points."""
        start_x = start[1] * self.block_size + (self.block_size / 2)
        start_y = -start[0] * self.block_size - (self.block_size / 2)
        end_x = end[1] * self.block_size + (self.block_size / 2)
        end_y = -end[0] * self.block_size - (self.block_size / 2)

        length = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
        angle = np.degrees(np.arctan2(end_y - start_y, end_x - start_x))
        flipped = angle < 0

        is_diagonal = abs(start[0] - end[0]) == 1 and abs(start[1] - end[1]) == 1
        support_width = (
            self.diagonal_support_width if is_diagonal else self.support_width
        )

        support = Box(
            length,
            support_width,
            self.block_height,
            align=(Align.CENTER, Align.CENTER, 0),
        )
        mid_x = start_x
        if is_diagonal:
            mid_y = (
                start_y
                + self.block_size
                # + (support_width / 2 if angle > 0 else -support_width / 2)
            )
            if flipped and start[1] < end[1]:
                mid_y -= self.block_size / 2
                mid_x += self.block_size / 2
                pass
            elif flipped and start[1] > end[1]:
                mid_y -= self.block_size / 2
                mid_x -= self.block_size / 2
                pass
        else:
            mid_y = start_y - support_width / 2
            # print(start, end, flipped)
            is_horizontal = start[0] == end[0]
            if is_horizontal:
                if start[1] < end[1]:
                    mid_x += length / 2
                else:
                    mid_x -= length / 2
                mid_y += length / 2
            else:
                # mid_x += length / 2
                pass
                if flipped:
                    # mid_y += 50
                    # mid_y += length / 2
                    pass
                else:
                    mid_y += length

        support.location = Location((mid_x, mid_y, 0), (0, 0, angle))
        return support

    def generate(self) -> Compound:
        """Generate the 3D QR code with comprehensive support structures."""
        blocks = []
        for row_index, row in enumerate(self.matrix):
            for col_index, module in enumerate(row):
                if module:
                    block = Box(
                        self.block_size,
                        self.block_size,
                        self.block_height,
                        align=(0, 0, 0),
                    )
                    block.location = Location(
                        (col_index * self.block_size, -row_index * self.block_size, 0)
                    )
                    blocks.append(block)

        # Create all necessary supports
        supports = []
        processed_pairs = set()

        # Add diagonal supports within components
        for row in range(self.rows):
            for col in range(self.cols):
                if self.matrix[row][col]:
                    diagonal_neighbors = self._needs_diagonal_support(row, col)
                    for neighbor in diagonal_neighbors:
                        pair = tuple(sorted([(row, col), neighbor]))
                        if pair not in processed_pairs:
                            support = self._create_support((row, col), neighbor)
                            supports.append(support)
                            processed_pairs.add(pair)

        # Add supports between components
        processed_components = set()
        for component_id in set(self.connected_components.values()):
            if component_id in processed_components:
                continue

            nearest_neighbors = self._find_nearest_neighbors(component_id)
            for start_pos, end_pos in nearest_neighbors:
                pair = tuple(sorted([start_pos, end_pos]))
                if pair not in processed_pairs:
                    support = self._create_support(start_pos, end_pos)
                    supports.append(support)
                    processed_pairs.add(pair)

            processed_components.add(component_id)

        return Compound.make_compound(blocks + supports)


# Example usage
if __name__ == "__main__":
    # this format enables phones to connect wifi by scanning the qr code
    wifi_name = "hello"
    wifi_password = "world"
    text = f"WIFI:S:{wifi_name};T:WPA2;P:{wifi_password};;"
    # text = "AA"
    generator = QRCode3DGenerator(
        text=text,
        block_size=5.0,
        block_height=4.0,
        support_width=0.8,
        diagonal_support_width=1.2,
    )
    qr_code_3d = generator.generate()
    show_object(qr_code_3d)

    # Export the 3D QR code to glTF format
    export_gltf(qr_code_3d, "qr_code_3d.gltf", binary=False)

    # TODO: improve the export time/format since this takes a long times
    # as it is encoding each block as a separate mesh, when it could be a single mesh
    export_3mf = False
    if export_3mf:
        start_export = time.time()
        exporter = Mesher()
        exporter.add_shape(qr_code_3d, part_number="qr_code_3d")
        exporter.add_code_to_metadata()
        exporter.write("qr_code_3d.3mf")
        print("Exporting 3mf file took", time.time() - start_export, "seconds")
