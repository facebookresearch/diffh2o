# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import trimesh

f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
        750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
f2 = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325, 326,
        327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
        355]

f3 = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
        439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467]
f4 = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
        550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578]
f5 = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
        668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
f0 = [73, 96, 98, 99, 772, 774, 775, 777]

class GraspGenMesh():
    def __init__(self,verts,faces,sampled_point,sampled_normal):
        # if verts is not a tensor

        self.verts = verts if torch.is_tensor(verts) else torch.tensor(verts,dtype=torch.float32).unsqueeze(0)

        self.faces = faces if torch.is_tensor(faces) else torch.tensor(faces,dtype=torch.int32).unsqueeze(0)

        self.sampled_point = sampled_point if torch.is_tensor(sampled_point) else torch.tensor(sampled_point,dtype=torch.float32).unsqueeze(0)

        self.sampled_normal = sampled_normal if torch.is_tensor(sampled_normal) else torch.tensor(sampled_normal,dtype=torch.float32).unsqueeze(0)

    def to(self,device):
        self.verts = self.verts.to(device)
        self.faces = self.faces.to(device)
        self.sampled_point = self.sampled_point.to(device)
        self.sampled_normal = self.sampled_normal.to(device)
        return self

    def repeat(self,n):
        self.verts = torch.repeat_interleave(self.verts,n,dim=0)
        self.faces = torch.repeat_interleave(self.faces,n,dim=0)
        self.sampled_point = torch.repeat_interleave(self.sampled_point,n,dim=0)
        self.sampled_normal = torch.repeat_interleave(self.sampled_normal,n,dim=0)
        return self

    def __len__(self):
        return self.verts.shape[0]

    def get_trimesh(self,idx):
        return trimesh.Trimesh(vertices=self.verts[idx].detach().cpu().numpy(),faces=self.faces[idx].detach().cpu().numpy())

    def remove_by_mask(self,mask):
        self.verts = self.verts[mask]
        self.faces = self.faces[mask]
        self.sampled_point = self.sampled_point[mask]
        self.sampled_normal = self.sampled_normal[mask]
        return self
