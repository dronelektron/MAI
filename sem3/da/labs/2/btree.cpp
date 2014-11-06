#include "btree.h"

TBTree::TBTree(size_t t)
{
	mT = t < 2 ? 2 : t;
	mRoot = mCreateNode();
}

TBTree::~TBTree()
{
	mDeleteTree(mRoot);
}

void TBTree::Insert(const TData& data)
{
	TNode* root = mRoot;

	if (root->n == mT * 2 - 1)
	{
		TNode* rootNew = mCreateNode();

		mRoot = rootNew;
		rootNew->leaf = false;
		rootNew->n = 0;
		rootNew->childs[0] = root;

		mSplitNode(rootNew, root, 0);
		mInsertNonFull(rootNew, data);
	}
	else
	{
		mInsertNonFull(root, data);
	}
}

TBTree::TData* TBTree::Find(const TKey& key)
{
	return mFind(mRoot, key);
}

void TBTree::Erase(const TKey& key)
{
	mErase(mRoot, key);

	if (!mRoot->leaf && mRoot->n == 0)
	{
		TNode* rootNew = mRoot->childs[0];

		mDeleteNode(mRoot);

		mRoot = rootNew;
	}
}

TBTree::TNode* TBTree::mCreateNode()
{
	TNode* node = NULL;

	try
	{
		node = new TNode;
		node->data = new TData[mT * 2 - 1];
		node->childs = new TNode*[mT * 2];
	}
	catch (const std::bad_alloc& e)
	{
		printf("ERROR: No memory\n");

		std::exit(0);
	}

	node->n = 0;
	node->leaf = true;

	size_t cnt = mT * 2;

	for (size_t i = 0; i < cnt; ++i)
	{
		node->childs[i] = NULL;
	}

	return node;
}

TBTree::TData* TBTree::mFind(TNode* node, const TKey& key)
{
	size_t i = 0;

	while (i < node->n && node->data[i].key < key)
	{
		++i;
	}

	if (i < node->n && node->data[i].key == key)
	{
		return &node->data[i];
	}

	if (!node->leaf)
	{
		return mFind(node->childs[i], key);
	}

	return NULL;
}

void TBTree::mDeleteNode(TNode* node)
{
	delete[] node->data;
	delete[] node->childs;
	delete node;
}

void TBTree::mDeleteTree(TNode* node)
{
	if (node == NULL)
	{
		return;
	}

	for (size_t i = 0; i < node->n + 1; ++i)
	{
		mDeleteTree(node->childs[i]);
	}
	
	mDeleteNode(node);
}

void TBTree::mShiftLeft(TNode* node, size_t index, size_t index2)
{
	for (size_t i = index; i < node->n; ++i)
	{
		node->data[i - 1].val = node->data[i].val;
		node->data[i - 1].key.Swap(node->data[i].key);
	}

	if (!node->leaf)
	{
		for (size_t i = index2; i < node->n + 1; ++i)
		{
			node->childs[i - 1] = node->childs[i];
		}
	}

	node->childs[node->n] = NULL;
	--node->n;
}

void TBTree::mShiftRight(TNode* node, size_t index, size_t index2)
{
	for (int i = static_cast<int>(node->n) - 1; i >= static_cast<int>(index); --i)
	{
		node->data[i + 1].val = node->data[i].val;
		node->data[i + 1].key.Swap(node->data[i].key);
	}

	if (!node->leaf)
	{
		for (int i = static_cast<int>(node->n); i >= static_cast<int>(index2); --i)
		{
			node->childs[i + 1] = node->childs[i];
		}
	}
	
	++node->n;
}

void TBTree::mFlowLeft(TNode* parent, size_t index)
{
	TNode* left = parent->childs[index];
	TNode* right = parent->childs[index + 1];

	left->data[left->n].val = parent->data[index].val;
	left->data[left->n].key.Swap(parent->data[index].key);

	parent->data[index].val = right->data[0].val;
	parent->data[index].key.Swap(right->data[0].key);

	if (!right->leaf)
	{
		left->childs[left->n + 1] = right->childs[0];
	}

	mShiftLeft(right, 1, 1);

	++left->n;
}

void TBTree::mFlowRight(TNode* parent, size_t index)
{
	TNode* left = parent->childs[index];
	TNode* right = parent->childs[index + 1];

	mShiftRight(right, 0, 0);

	right->data[0].val = parent->data[index].val;
	right->data[0].key.Swap(parent->data[index].key);
	
	parent->data[index].val = left->data[left->n - 1].val;
	parent->data[index].key.Swap(left->data[left->n - 1].key);

	if (!left->leaf)
	{
		right->childs[0] = left->childs[left->n];
	}

	left->childs[left->n] = NULL;
	--left->n;
}

void TBTree::mSplitNode(TNode* parent, TNode* node, size_t index)
{
	TNode* node2 = mCreateNode();

	node2->leaf = node->leaf;
	node2->n = mT - 1;
	node->n = node2->n;

	for (size_t i = 0; i < node2->n; ++i)
	{
		node2->data[i].val = node->data[i + mT].val;
		node2->data[i].key.Swap(node->data[i + mT].key);
	}
	
	if (!node->leaf)
	{
		for (size_t i = 0; i < mT; ++i)
		{
			node2->childs[i] = node->childs[i + mT];
		}
	}

	mShiftRight(parent, index, index + 1);

	parent->childs[index + 1] = node2;
	
	parent->data[index].val = node->data[mT - 1].val;
	parent->data[index].key.Swap(node->data[mT - 1].key);
}

void TBTree::mMergeNode(TNode* parent, size_t index)
{
	TNode* left = parent->childs[index];
	TNode* right = parent->childs[index + 1];

	left->data[left->n].val = parent->data[index].val;
	left->data[left->n].key.Swap(parent->data[index].key);

	for (size_t i = 0; i < right->n; ++i)
	{
		left->data[left->n + i + 1].val = right->data[i].val;
		left->data[left->n + i + 1].key.Swap(right->data[i].key);
	}

	if (!right->leaf)
	{
		for (size_t i = 0; i < right->n + 1; ++i)
		{
			left->childs[left->n + i + 1] = right->childs[i];
		}
	}

	left->n += right->n + 1;

	mShiftLeft(parent, index + 1, index + 2);
	mDeleteNode(right);
}

void TBTree::mInsertNonFull(TNode* node, const TData& data)
{
	int i = static_cast<int>(node->n) - 1;

	if (node->leaf)
	{
		while (i >= 0 && data.key < node->data[i].key)
		{
			node->data[i + 1].val = node->data[i].val;
			node->data[i + 1].key.Swap(node->data[i].key);
			--i;
		}

		node->data[i + 1] = data;
		++node->n;
	}
	else
	{
		while (i >= 0 && data.key < node->data[i].key)
		{
			--i;
		}

		++i;

		if (node->childs[i]->n == mT * 2 - 1)
		{
			mSplitNode(node, node->childs[i], i);

			if (node->data[i].key < data.key)
			{
				++i;
			}
		}

		mInsertNonFull(node->childs[i], data);
	}
}

void TBTree::mErase(TNode* node, const TKey& key)
{
	size_t i = 0;

	while (i < node->n && node->data[i].key < key)
	{
		++i;
	}

	if (i < node->n && node->data[i].key == key)
	{
		if (node->leaf)
		{
			mShiftLeft(node, i + 1, i + 2);
		}
		else
		{
			TNode* succNode = node->childs[i + 1];

			while (succNode->childs[0] != NULL)
			{
				succNode = succNode->childs[0];
			}

			node->data[i].val = succNode->data[0].val;
			node->data[i].key.Swap(succNode->data[0].key);

			mErase(node->childs[i + 1], succNode->data[0].key);
			mFixChild(node, i + 1);
		}
	}
	else if (!node->leaf)
	{
		mErase(node->childs[i], key);
		mFixChild(node, i);
	}
}

void TBTree::mFixChild(TNode* node, size_t index)
{
	if (node->childs[index]->n >= mT)
	{
		return;
	}

	TNode* left = index > 0 ? node->childs[index - 1] : NULL;
	TNode* right = index < node->n ? node->childs[index + 1] : NULL;

	if (left != NULL && right != NULL)
	{
		if (left->n >= mT)
		{
			mFlowRight(node, index - 1);
		}
		else if (right->n >= mT)
		{
			mFlowLeft(node, index);
		}
		else
		{
			mMergeNode(node, index - 1);
		}
	}
	else if (left != NULL)
	{
		if (left->n >= mT)
		{
			mFlowRight(node, index - 1);
		}
		else
		{
			mMergeNode(node, index - 1);
		}
	}
	else
	{
		if (right->n >= mT)
		{
			mFlowLeft(node, index);
		}
		else
		{
			mMergeNode(node, index);
		}
	}
}

bool TBTree::Serialize(const char* filename)
{
	FILE* f = fopen(filename, "wb");

	if (f == NULL)
	{
		return false;
	}

	if (fwrite(&mT, sizeof(mT), 1, f) != 1)
	{
		return false;
	}

	bool ans = mSerialize(f, mRoot);

	fclose(f);

	return ans;
}

bool TBTree::Deserialize(const char* filename)
{
	FILE* f = fopen(filename, "rb");

	if (f == NULL)
	{
		return false;
	}

	if (fread(&mT, sizeof(mT), 1, f) != 1)
	{
		return false;
	}

	TNode* rootNew = mCreateNode();
	bool ans = mDeserialize(f, rootNew);

	fclose(f);

	if (ans)
	{
		mDeleteTree(mRoot);

		mRoot = rootNew;

		return true;
	}
	else
	{
		mDeleteTree(rootNew);

		return false;
	}
}

bool TBTree::mSerialize(FILE* f, TNode* node)
{
	if (fwrite(&node->n, sizeof(node->n), 1, f) != 1)
	{
		return false;
	}

	if (fwrite(&node->leaf, sizeof(node->leaf), 1, f) != 1)
	{
		return false;
	}

	for (size_t i = 0; i < node->n; ++i)
	{
		const TData* data = &node->data[i];
		const size_t strLen = data->key.Length();
		const char* str = data->key.Str();

		if (fwrite(&strLen, sizeof(strLen), 1, f) != 1)
		{
			return false;
		}

		if (fwrite(str, sizeof(char), strLen, f) != strLen)
		{
			return false;
		}

		if (fwrite(&data->val, sizeof(data->val), 1, f) != 1)
		{
			return false;
		}
	}

	if (!node->leaf)
	{
		for (size_t i = 0; i < node->n + 1; ++i)
		{
			if (!mSerialize(f, node->childs[i]))
			{
				return false;
			}
		}
	}

	return true;
}

bool TBTree::mDeserialize(FILE* f, TNode* node)
{
	char buffer[257];

	if (fread(&node->n, sizeof(node->n), 1, f) != 1)
	{
		return false;
	}

	if (fread(&node->leaf, sizeof(node->leaf), 1, f) != 1)
	{
		return false;
	}

	for (size_t i = 0; i < node->n; ++i)
	{
		TData* data = &node->data[i];
		size_t strLen = 0;

		if (fread(&strLen, sizeof(strLen), 1, f) != 1)
		{
			return false;
		}

		if (fread(buffer, sizeof(char), strLen, f) != strLen)
		{
			return false;
		}

		if (fread(&data->val, sizeof(data->val), 1, f) != 1)
		{
			return false;
		}

		buffer[strLen] = '\0';
		data->key = buffer;
	}

	if (!node->leaf)
	{
		for (size_t i = 0; i < node->n + 1; ++i)
		{
			node->childs[i] = mCreateNode();

			if (!mDeserialize(f, node->childs[i]))
			{
				return false;
			}
		}
	}

	return true;
}
