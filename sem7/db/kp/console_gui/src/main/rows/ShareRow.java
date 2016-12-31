package main.rows;

public class ShareRow {
	public ShareRow(int shareId, int userId, int storageId, int accessId) {
		this.shareId = shareId;
		this.userId = userId;
		this.storageId = storageId;
		this.accessId = accessId;
	}

	public int getShareId() {
		return shareId;
	}

	public int getUserId() {
		return userId;
	}

	public int getStorageId() {
		return storageId;
	}

	public int getAccessId() {
		return accessId;
	}

	public void setUserId(int userId) {
		this.userId = userId;
	}

	public void setStorageId(int storageId) {
		this.storageId = storageId;
	}

	public void setAccessId(int accessId) {
		this.accessId = accessId;
	}

	private int shareId;
	private int userId;
	private int storageId;
	private int accessId;
}
