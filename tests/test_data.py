import os
import unittest
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader

from trunk_sim.data import TrunkData, TrunkTorchDataset


class TestTrunkData(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a basic TrunkData object for position states of top 2 links
        self.num_links = 3
        self.data_l12 = TrunkData(states="pos", links=[1, 2], num_links=self.num_links)

        # Create a TrunkData object for position and velocity states of only third link
        self.data_vel_l3 = TrunkData(
            states="pos_vel", links=[3], num_links=self.num_links
        )

        # Create a TrunkData object for velocity states of 3 links
        self.data_vel_l123 = TrunkData(
            states="vel", links=[1, 2, 3], num_links=self.num_links
        )

        # Create some sample data
        self.t = 1.0
        self.x = np.array(
            [
                [0.1, 0.2, 0.3] * 2,  # x,y,z,vx,vy,vz for first link
                [0.4, 0.5, 0.6] * 2,  # x,y,z,vx,vy,vz for second link
                [0.7, 0.8, 0.9] * 2,
            ]
        )  # x,y,z,vx,vy,vz for third link
        self.u = np.array(
            [
                [1.0, 2.0],  # ux,uy for first link
                [3.0, 4.0],  # ux,uy for second link
                [5.0, 6.0],
            ]
        )  # ux,uy for third link
        self.x_new = np.array(
            [
                [0.4, 0.5, 0.6] * 2,  # x,y,z,vx,vy,vz for first link
                [0.7, 0.8, 0.9] * 2,  # x,y,z,vx,vy,vz for second link
                [1.0, 1.1, 1.2] * 2,
            ]
        )  # x,y,z,vx,vy,vz for third link

        self.t_batch = np.array([1.0, 2.0, 3.0])
        self.x_batch = 0.1 * np.arange(
            len(self.t_batch) * np.prod(self.x.shape)
        ).reshape(len(self.t_batch), *self.x.shape)
        self.u_batch = np.arange(len(self.t_batch) * np.prod(self.u.shape)).reshape(
            len(self.t_batch), *self.u.shape
        )
        self.x_new_batch = 0.1 * np.arange(
            3, len(self.t_batch) * np.prod(self.x.shape) + 3
        ).reshape(len(self.t_batch), *self.x_new.shape)

    def test_initialization(self):
        """
        Test proper initialization with different state types.
        """
        # Test position states
        self.assertEqual(self.data_l12.states, "pos")
        self.assertEqual(self.data_l12.links, [1, 2])
        self.assertEqual(self.data_l12.state_cols, ["x1", "y1", "z1", "x2", "y2", "z2"])
        self.assertEqual(
            self.data_l12.control_cols, ["ux1", "uy1", "ux2", "uy2", "ux3", "uy3"]
        )
        self.assertEqual(self.data_l12.state_dim, 6)
        self.assertEqual(self.data_l12.control_dim, 6)

        # Test position-velocity states
        self.assertEqual(self.data_vel_l3.states, "pos_vel")
        self.assertEqual(self.data_vel_l3.links, [3])
        self.assertEqual(
            self.data_vel_l3.state_cols, ["x3", "y3", "z3", "vx3", "vy3", "vz3"]
        )
        self.assertEqual(self.data_vel_l3.state_dim, 6)

        # Test velocity states
        self.assertEqual(self.data_vel_l123.states, "vel")
        self.assertEqual(self.data_vel_l123.links, [1, 2, 3])
        self.assertEqual(
            self.data_vel_l123.state_cols,
            ["vx1", "vy1", "vz1", "vx2", "vy2", "vz2", "vx3", "vy3", "vz3"],
        )
        self.assertEqual(self.data_vel_l123.state_dim, 9)

    def test_add_data(self):
        """
        Test adding a single data point.
        """
        self.data_l12.add_data(self.t, self.x, self.u, self.x_new)

        # Check that data was added correctly
        self.assertEqual(len(self.data_l12), 1)
        self.assertEqual(self.data_l12.dataframe.iloc[0]["t"], self.t)

        # Check each state value
        for i, col in enumerate(self.data_l12.state_cols):
            self.assertEqual(
                self.data_l12.dataframe.iloc[0][col],
                self.x[i // self.num_links, i % self.num_links],
            )

        # Check each control input value
        for i, col in enumerate(self.data_l12.control_cols):
            self.assertEqual(
                self.data_l12.dataframe.iloc[0][col], self.u[i // 2, i % 2]
            )

    def test_add_data_wrong_dimensions(self):
        """
        Test adding data with wrong dimensions raises an assertion error.
        """
        # Both x and u have too few elements
        wrong_x = np.array([[0.1, 0.2], [0.3, 0.4]])
        wrong_u = np.array([[1.0], [2.0]])
        wrong_x_new = np.array([[0.1, 0.2], [0.3, 0.4]])

        with self.assertRaises(AssertionError):
            self.data_l12.add_data(self.t, wrong_x, self.u, self.x_new)

        with self.assertRaises(AssertionError):
            self.data_l12.add_data(self.t, self.x, wrong_u, self.x_new)

        with self.assertRaises(AssertionError):
            self.data_l12.add_data(self.t, self.x, self.u, wrong_x_new)

        with self.assertRaises(AssertionError):
            self.data_l12.add_data(self.t, wrong_x, wrong_u, wrong_x_new)

    def test_add_batch_data(self):
        """
        Test adding a batch of data points.
        """
        self.data_l12.add_batch_data(
            self.t_batch, self.x_batch, self.u_batch, self.x_new_batch
        )

        # Check that all data was added correctly
        self.assertEqual(len(self.data_l12), len(self.t_batch))

        # Check values for each time step
        for i in range(len(self.t_batch)):
            # Check time
            self.assertEqual(self.data_l12.dataframe.iloc[i]["t"], self.t_batch[i])

            # Check each state value
            for j, col in enumerate(self.data_l12.state_cols):
                self.assertEqual(
                    self.data_l12.dataframe.iloc[i][col],
                    self.x_batch[i, j // self.num_links, j % self.num_links],
                )

            # Check each control input value
            for j, col in enumerate(self.data_l12.control_cols):
                self.assertEqual(
                    self.data_l12.dataframe.iloc[i][col], self.u_batch[i, j // 2, j % 2]
                )

            # Check each new state value
            for j, col in enumerate(self.data_l12.state_new_cols):
                self.assertEqual(
                    self.data_l12.dataframe.iloc[i][col],
                    self.x_new_batch[i, j // self.num_links, j % self.num_links],
                )

    def test_save_and_load_csv(self):
        """
        Test saving and loading data from a CSV file.
        """
        # Add some data first
        self.data_l12.add_data(self.t, self.x, self.u, self.x_new)
        self.data_l12.add_batch_data(
            self.t_batch, self.x_batch, self.u_batch, self.x_new_batch
        )

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save to CSV
            self.data_l12.save_to_csv(tmp_path)

            # Check that the file exists and has content
            self.assertTrue(os.path.exists(tmp_path))
            self.assertTrue(os.path.getsize(tmp_path) > 0)

            # Create a new TrunkData object and load the CSV
            new_data = TrunkData(states="pos", links=[1, 2], num_links=self.num_links)
            new_data.load_from_csv(tmp_path)

            # Verify the loaded data
            self.assertEqual(len(new_data), 4)  # 1 single point + 3 batch points

            # Check some values to ensure data was loaded correctly
            self.assertEqual(new_data.dataframe.iloc[0]["t"], self.t)
            self.assertEqual(new_data.dataframe.iloc[1]["t"], self.t_batch[0])

            # Check a few state values
            self.assertEqual(new_data.dataframe.iloc[0]["x1"], self.x[0, 0])
            self.assertEqual(new_data.dataframe.iloc[3]["z2"], self.x_batch[2, 1, 2])

        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_convert_to_torch_dataset(self):
        """
        Test converting to a PyTorch dataset.
        """
        # Add some data
        self.data_l12.add_batch_data(
            self.t_batch, self.x_batch, self.u_batch, self.x_new_batch
        )

        # Convert to torch dataset
        torch_dataset = self.data_l12.convert_to_torch_dataset()

        # Check dataset properties
        self.assertIsInstance(torch_dataset, TrunkTorchDataset)
        self.assertEqual(len(torch_dataset), 3)

        # Check dataset items
        inputs, outputs = torch_dataset[0]
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(outputs, torch.Tensor)

        # Check tensor shapes
        self.assertEqual(
            inputs.shape[0], self.data_l12.state_dim + self.data_l12.control_dim
        )
        self.assertEqual(outputs.shape[0], self.data_l12.state_dim)

        # Check values
        np.testing.assert_array_equal(
            inputs.numpy().astype(np.float32),
            np.concat(
                [self.x_batch[0, :2, :3].flatten(), self.u_batch[0].flatten()]
            ).astype(np.float32),
        )
        np.testing.assert_array_equal(
            outputs.numpy().astype(np.float32),
            self.x_new_batch[0, :2, :3].flatten().astype(np.float32),
        )

        # Test creating a data loader
        dataloader = DataLoader(torch_dataset, batch_size=2, shuffle=False)
        for batch_inputs, batch_outputs in dataloader:
            self.assertEqual(
                batch_inputs.shape,
                (2, self.data_l12.state_dim + self.data_l12.control_dim),
            )
            self.assertEqual(batch_outputs.shape, (2, self.data_l12.state_dim))
            break

    def test_custom_convert_to_torch_dataset(self):
        """
        Test converting to a PyTorch dataset with custom column selections.
        """
        # Add some data
        self.data_l12.add_batch_data(
            self.t_batch, self.x_batch, self.u_batch, self.x_new_batch
        )

        # Convert to torch dataset with custom columns
        input_cols = ["x1", "y1", "z1"]
        output_cols = ["ux1", "uy1"]
        torch_dataset = self.data_l12.convert_to_torch_dataset(
            input_cols=input_cols, output_cols=output_cols
        )

        # Check dataset properties
        self.assertEqual(len(torch_dataset), 3)

        # Check tensor shapes with custom columns
        inputs, outputs = torch_dataset[0]
        self.assertEqual(inputs.shape[0], len(input_cols))
        self.assertEqual(outputs.shape[0], len(output_cols))

        # Check specific values
        np.testing.assert_array_equal(
            inputs.numpy().astype(np.float32),
            self.x_batch[0, 0, :3].flatten().astype(np.float32),
        )
        np.testing.assert_array_equal(
            outputs.numpy().astype(np.float32),
            self.u_batch[0, 0].flatten().astype(np.float32),
        )

    def test_get_data_at_time(self):
        """
        Test retrieving row of data at a specific time.
        """
        # Add some data
        self.data_l12.add_batch_data(
            self.t_batch, self.x_batch, self.u_batch, self.x_new_batch
        )

        # Get state at exact time
        data_row = self.data_l12.get_data_at_time(2.0)
        self.assertEqual(data_row[0], 2.0)
        np.testing.assert_array_equal(
            data_row[1:],
            np.concat(
                [
                    self.x_batch[1, :2, :3].flatten(),
                    self.u_batch[1].flatten(),
                    self.x_new_batch[1, :2, :3].flatten(),
                ]
            ),
        )

        # Get state at time between points (should get closest)
        data_row = self.data_l12.get_data_at_time(2.1)
        np.testing.assert_array_equal(
            data_row,
            np.concat(
                [
                    np.array([2.0]),
                    self.x_batch[1, :2, :3].flatten(),
                    self.u_batch[1].flatten(),
                    self.x_new_batch[1, :2, :3].flatten(),
                ]
            ),
        )
        data_row = self.data_l12.get_data_at_time(2.6)
        np.testing.assert_array_equal(
            data_row,
            np.concat(
                [
                    np.array([3.0]),
                    self.x_batch[2, :2, :3].flatten(),
                    self.u_batch[2].flatten(),
                    self.x_new_batch[2, :2, :3].flatten(),
                ]
            ),
        )

    def test_len_and_getitem(self):
        """
        Test the __len__ and __getitem__ methods.
        """
        # Add some data
        self.data_l12.add_batch_data(
            self.t_batch, self.x_batch, self.u_batch, self.x_new_batch
        )

        # Test __len__
        self.assertEqual(len(self.data_l12), 3)

        # Test __getitem__
        row = self.data_l12[1]
        self.assertEqual(row["t"], self.t_batch[1])
        self.assertEqual(row["x1"], self.x_batch[1, 0, 0])
        self.assertEqual(row["ux2"], self.u_batch[1, 1, 0])


if __name__ == "__main__":
    unittest.main()
